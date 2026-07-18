from pathlib import Path
from typing import Any

import numpy as np
import schedulefree
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric

from cpgpt.loss.loss import (
    beta_loss,
    consistency_loss,
    contrastive_loss,
    kld_bernoulli_loss,
    kld_normal_loss,
    wd_loss,
)

from .utils import (
    SaveOutput,
    beta_to_m,
    m_to_beta,
    patch_attention,
    safe_clone,
)


class CpGPTLitModule(LightningModule):
    """A LightningModule for CpG site prediction using deep learning.

    This module implements a deep learning model for predicting CpG methylation sites,
    supporting various training configurations including condition prediction.

    Args:
        net (torch.nn.Module): Neural network model for CpG prediction.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        training (Dict[str, Any]): Training configuration dictionary.
        compile (bool): Whether to compile the model using torch.compile.
        **kwargs: Additional keyword arguments passed to parent class.

    Attributes:
        net (torch.nn.Module): The neural network model.
        train_loss (MeanMetric): Metric for tracking training loss.
        val_loss (MeanMetric): Metric for tracking validation loss.
        test_loss (MeanMetric): Metric for tracking test loss.
        val_loss_best (MinMetric): Metric for tracking best validation loss.

    """

    def __init__(
        self,
        training: dict[str, Any],
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        compile: bool = False,
        seed: int = 42,
    ) -> None:
        """Initialize the CpGPTLitModule.

        Args:
            training (Dict[str, Any]): Training configuration dictionary.
            net (torch.nn.Module): Neural network model for CpG prediction.
            optimizer (torch.optim.Optimizer): Optimizer for model training.
            scheduler (Optional[torch.optim.lr_scheduler.LRScheduler]): Learning rate scheduler.
            compile (bool): Whether to compile the model using torch.compile.
            seed: Seed for random number generation.

        Note:
            The module initializes various metrics for tracking training progress and
            supports different training-time behaviors based on the training configuration.

        """
        super().__init__()
        self.automatic_optimization = False

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation loss
        self.val_loss_best = MinMetric()

        # for random number generation
        self.rng = np.random.default_rng(seed)

        # Diffusion training is intentionally not supported.

    def setup(self, stage: str) -> None:
        """Set up the model for a specific stage of training.

        This method is called by Lightning before training/validation/testing.
        Compiles the model if specified in the configuration.

        Args:
            stage (str): Stage of training ('fit', 'validate', 'test', or 'predict').

        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizers and learning rate schedulers.

        Returns:
            Dict[str, Any]: Configuration dictionary containing optimizer and optional scheduler.

        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if "scheduler" in self.hparams and not isinstance(
            optimizer,
            schedulefree.AdamWScheduleFree,
        ):
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def on_train_start(self) -> None:
        """Initialize training by resetting validation metrics.

        This method is called at the start of training and ensures validation
        metrics don't retain results from sanity checks.
        """
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_loss_best.reset()

    def on_validation_epoch_end(self) -> None:
        """Process validation results at the end of an epoch.

        Computes the current validation loss, updates the best validation loss,
        and logs metrics.
        """
        loss = self.val_loss.compute()  # get current val loss
        self.val_loss_best(loss)  # update best so far val loss
        # log `val_loss_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)
        self.val_loss.reset()  # reset val loss for next epoch

    def on_predict_epoch_start(self) -> None:
        """Initialize embeddings at the start of prediction epoch if in reconstruct mode."""
        predict_mode = getattr(self, "predict_mode_predict", "forward")
        n_thinking_steps = getattr(self, "n_thinking_steps_predict", 0)

        if predict_mode != "reconstruct":
            return

        datamodule = self.trainer.datamodule
        if datamodule is None:
            msg = "No datamodule is attached to the trainer."
            raise ValueError(msg)

        species = getattr(self, "species_predict", None)
        if species is None:
            msg = "No species provided via kwargs or datamodule hparams."
            raise ValueError(msg)

        embedder = getattr(datamodule, "embedder", None)
        if embedder is None:
            msg = "No embedder found in the datamodule."
            raise ValueError(msg)

        dna_llm = datamodule.hparams.dna_llm
        dna_context_len = datamodule.hparams.dna_context_len

        if species not in embedder.ensembl_metadata_dict:
            msg = f"Species {species} not found in the embedder."
            raise ValueError(msg)
        if dna_llm not in embedder.ensembl_metadata_dict[species]:
            msg = f"DNA LLM {dna_llm} not found in the embedder for species {species}."
            raise ValueError(msg)

        embeddings_file = (
            Path(embedder.dependencies_dir)
            / "dna_embeddings"
            / species
            / dna_llm
            / f"{dna_context_len}bp_dna_embeddings.mmap"
        )
        embeddings_loc_dict = embedder.ensembl_metadata_dict[species][dna_llm][dna_context_len]

        # Pre-load genomic locations
        genomic_locations = getattr(self, "genomic_locations_predict", None)
        if genomic_locations is None:
            msg = "For reconstruction mode, 'genomic_locations' must be provided as a kwarg."
            raise ValueError(
                msg,
            )

        # Pre-compute embedding indices
        embedding_indices = []
        for loc in genomic_locations:
            if loc not in embeddings_loc_dict:
                msg = f"Location {loc} not found in embeddings"
                raise ValueError(msg)
            embedding_indices.append(embeddings_loc_dict[loc])

        # Load and convert embeddings to tensor once
        embeddings = np.memmap(
            embeddings_file,
            dtype="float32",
            mode="r",
            shape=(len(embeddings_loc_dict), embedder.llm_embedding_size_dict[dna_llm]),
        )

        # Cache embeddings
        self.cached_embeddings = torch.tensor(
            embeddings[embedding_indices],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        if n_thinking_steps == 0:
            return

        if datamodule.hparams.batch_size > 1:
            msg = "Chain of thought (n_thinking_steps > 0) requires batch_size=1"
            raise ValueError(msg)

        # Retrieve all available genomic locations
        all_genomic_locations = sorted(embeddings_loc_dict.keys())

        # Convert genomic locations to positions and chromosomes
        positions = [int(loc.split(":")[1]) for loc in all_genomic_locations]
        chromosomes = [loc.split(":")[0].replace("chr", "") for loc in all_genomic_locations]
        chromosomes = [
            embedder.ensembl_metadata_dict[species]["vocab"][chrom] for chrom in chromosomes
        ]

        # Store genomic locations, positions, and chromosomes
        self.genomic_locations_predict = all_genomic_locations
        self.positions_predict = torch.tensor(
            positions, device=self.device, dtype=torch.int32
        ).unsqueeze(0)
        self.chromosomes_predict = torch.tensor(
            chromosomes, device=self.device, dtype=torch.int32
        ).unsqueeze(0)

        # Pre-compute embedding indices
        all_embedding_indices = [embeddings_loc_dict[loc] for loc in all_genomic_locations]

        # Cache embeddings
        self.all_cached_embeddings = torch.tensor(
            embeddings[all_embedding_indices],
            dtype=torch.float16,
            device=self.device,
        ).unsqueeze(0)

        # Clean up memmap
        del embeddings

    def on_predict_epoch_end(self) -> None:
        """Clean up embeddings at the end of prediction epoch."""
        if hasattr(self, "cached_embeddings"):
            del self.cached_embeddings
        if hasattr(self, "all_cached_embeddings"):
            del self.all_cached_embeddings

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Execute a single training step.

        Processes a batch of data through the model, computes losses, and accumulates
        gradients. Performs optimization step when accumulation cycle is complete.

        Args:
            batch (Dict[str, Any]): Input batch containing methylation data and metadata.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Computed loss value.

        """
        # Step 1: Setup optimization components
        # 1.1: Retrieve optimizer and scheduler
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers() if self.trainer.lr_scheduler_configs else None

        # 1.2: If optimizer is schedulefree, explicitly set it to train mode
        if isinstance(optimizer, schedulefree.AdamWScheduleFree):
            optimizer.train()

        # 1.3: Get gradient accumulation steps from trainer
        accumulate_grad_batches = self.trainer.accumulate_grad_batches

        # 1.4: Zero gradients only at the start of an accumulation cycle
        if batch_idx % accumulate_grad_batches == 0:
            optimizer.zero_grad()

        # Step 2: Prepare data and initialize variables
        # 2.1: Get the number of splits from hyperparameters
        n_splits = self.hparams.training.get("generative_splits", 2)

        # 2.2: Preprocess methylation data and NA mask
        meth, mask_na = self._get_meth_and_na_mask(batch)
        meth_original = safe_clone(meth)

        # 2.3: Get batch size and training progress information
        batch_size, _ = meth.shape
        total_steps = self.trainer.max_steps
        current_step = self.global_step

        # 2.4: Initialize input masks for the generative process
        current_input_masks = self._initialize_input_masks(meth, mask_na, n_splits)
        previous_input_masks = safe_clone(current_input_masks)

        # Step 3: Compute full data embedding for consistency loss (if enabled)
        if self.hparams.training["loss_weights"].get("consistency", 0.0) > 0.0:
            # 3.1: Prepare input data with all available methylation sites
            input_data = self._pad_input_data(
                batch,
                meth,
                mask_na,
                ~mask_na,
                binarize_input=self.hparams.training.get("binarize_input", False),
            )

            # 3.2: Encode DNA sequence
            full_sequence_embeddings = self.net.encode_sequence(batch["dna_embeddings"])

            # 3.3: Forward pass to get sample embedding with full data
            loss_inputs = self._forward_pass(
                batch=batch,
                input_data=input_data,
                loss_inputs={},
                full_sequence_embeddings=full_sequence_embeddings,
                current_input_masks=~mask_na,
            )

            # 3.4: Store the full sample embedding for consistency loss
            sample_embedding_full = loss_inputs["sample_embedding"]

        # Step 4: Iterative training loop over splits
        for step in range(1, max(n_splits, 2)):
            # 4.1: Prepare input data for current split
            input_data = self._pad_input_data(
                batch,
                meth,
                mask_na,
                current_input_masks,
                binarize_input=self.hparams.training.get("binarize_input", False),
            )

            # 4.2: Encode DNA sequence for current split
            full_sequence_embeddings = self.net.encode_sequence(batch["dna_embeddings"])

            # 4.3: Initialize loss inputs with original methylation data
            loss_inputs = {"meth": meth_original, "mask_na": mask_na}

            # 4.4: Add full sample embedding if consistency loss is enabled
            if self.hparams.training["loss_weights"].get("consistency", 0.0) > 0.0:
                loss_inputs["sample_embedding_full"] = sample_embedding_full

            # 4.5: Forward pass with current input masks
            loss_inputs = self._forward_pass(
                batch=batch,
                input_data=input_data,
                loss_inputs=loss_inputs,
                full_sequence_embeddings=full_sequence_embeddings,
                current_input_masks=current_input_masks,
            )

            # 4.6: Calculate all losses
            losses = self._calculate_losses(**loss_inputs)

            # 4.7: Scale loss by accumulation steps to maintain stable loss values
            normalized_loss = losses["loss"] / accumulate_grad_batches

            # 4.8: Accumulate gradients
            self.manual_backward(normalized_loss)

            # 4.9: Update input masks and methylation data for next split (if not the last split)
            if step < n_splits - 1:
                # 4.9.1: Update methylation values and input masks based on model predictions
                meth, current_input_masks = self._update_input_masks_and_meth(
                    meth=meth,
                    meth_original=meth_original,
                    mask_na=mask_na,
                    loss_inputs=loss_inputs,
                    current_input_masks=current_input_masks,
                    previous_input_masks=previous_input_masks,
                    n_splits=n_splits,
                    total_steps=total_steps,
                    current_step=current_step,
                )

                # 4.9.2: Update previous input masks for next iteration
                previous_input_masks = safe_clone(current_input_masks)

        # Step 5: Perform optimization step
        # 5.1: Check if we should perform optimizer step (end of accumulation or epoch)
        is_last_in_epoch = (batch_idx + 1) == len(self.trainer.train_dataloader)
        if (batch_idx + 1) % accumulate_grad_batches == 0 or is_last_in_epoch:
            # 5.2: Clip gradients to prevent exploding gradients
            self.clip_gradients(optimizer, gradient_clip_val=2.0, gradient_clip_algorithm="norm")

            # 5.3: Update model parameters
            optimizer.step()

            # 5.4: Update learning rate if scheduler is available
            if scheduler is not None:
                scheduler.step()

        # Step 6: Log training metrics
        self._log_training_metrics(losses, batch_size, optimizer)

        # Step 7: Return the total loss
        return losses["loss"]

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Perform a single validation step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
                Input batch tuple
            batch_idx (int): Batch index

        """
        loss_dict = self._shared_val_step(batch)
        loss = loss_dict["loss"]

        # update and log metrics
        self.val_loss(loss)
        for key, value in loss_dict.items():
            self.log(f"val/{key}", value, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        """Perform a single test step.

        Args:
            batch (Dict[str, Any]): Input batch dictionary
            batch_idx (int): Batch index

        """
        loss_dict = self._shared_val_step(batch)
        loss = loss_dict["loss"]

        # update and log metrics
        self.test_loss(loss)
        for key, value in loss_dict.items():
            self.log(f"test/{key}", value, on_step=False, on_epoch=True, prog_bar=True)

    def _shared_val_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Perform a shared validation step.

        Args:
            batch (Dict[str, Any]): A dictionary containing the input batch data

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the calculated losses

        """
        # Retrieve optimizer
        optimizer = self.optimizers()

        # If optimizer is schedulefree, we need to explicitly set it to eval mode
        if isinstance(optimizer, schedulefree.AdamWScheduleFree):
            optimizer.eval()

        # Use 1 or 2 splits for validation
        n_splits = min(self.hparams.training.get("generative_splits", 2), 2)

        # Step 1: Preprocess Data
        meth, mask_na = self._get_meth_and_na_mask(batch)
        meth_original = safe_clone(meth)

        batch_size, n_sites = meth.shape

        # Step 2: Initialize Input Masks
        current_input_masks = self._initialize_input_masks(meth, mask_na, n_splits)

        # Step 3: Prepare Input Data
        input_data = self._pad_input_data(
            batch,
            meth,
            mask_na,
            current_input_masks,
            binarize_input=self.hparams.training.get("binarize_input", False),
        )

        # Step 4: Forward Pass
        full_sequence_embeddings = self.net.encode_sequence(batch["dna_embeddings"])
        loss_inputs = {"meth": meth_original, "mask_na": mask_na}
        loss_inputs = self._forward_pass(
            batch,
            input_data,
            loss_inputs,
            full_sequence_embeddings,
            current_input_masks,
        )

        # Step 5: Calculate Losses
        return self._calculate_losses(**loss_inputs)

    def predict_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
        **kwargs: str | float | bool | list[str] | None,
    ) -> dict[str, torch.Tensor]:
        """Generate predictions for a batch of data.

        The prediction mode is determined by the `predict_mode` kwarg passed to trainer.predict().
        Supported modes:
        - "forward": Default forward step.
        - "attention": Extract attention weights.
        - "reconstruct": Reconstruct methylation values from given genomic locations.

        Additionally, `return_keys` can be passed to filter the keys from the output dictionary.

        Args:
            batch (Dict[str, Any]): Input batch containing data for prediction.
            batch_idx (int): Index of the current batch.
            dataloader_idx (int): Index of the current dataloader in multi-dataloader setups.
            **kwargs: Additional keyword arguments that may include:
                - predict_mode (str): "forward", "attention", or "reconstruct". Defaults to
                    "forward".
                - return_keys (List[str]): Keys to return from the output dictionary
                - layer_index (int): Layer index for attention mode
                - aggregate_heads (str): Method to aggregate attention heads
                - species (str): Species name for reconstruction mode
                - genomic_locations (List[str]): Locations for reconstruction mode
                - n_splits (int): Number of splits for reconstruction mode. Defaults to 1.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing predictions depending on the mode,
            filtered by `return_keys` if provided.

        """
        predict_mode = getattr(self, "predict_mode_predict", "forward")
        return_keys = getattr(self, "return_keys_predict", None)
        n_splits = getattr(self, "n_splits_predict", 1)

        if predict_mode == "attention":
            output_dictionary = self._predict_attention_step(batch, **kwargs)
        elif predict_mode == "reconstruct":
            output_dictionary = self._predict_reconstruct_step(batch, **kwargs)
        else:
            output_dictionary = self._predict_forward_step(batch, n_splits)

        if return_keys is not None:
            # Filter the dictionary to include only keys specified in return_keys
            output_dictionary = {k: v for k, v in output_dictionary.items() if k in return_keys}

        return output_dictionary

    def _predict_forward_step(
        self, batch: dict[str, Any], n_splits: int
    ) -> dict[str, torch.Tensor]:
        """Perform the default forward prediction step.

        This step executes a forward pass using the provided batch data and returns
        predicted values.

        Args:
            batch (Dict[str, Any]): Input batch for prediction.
            n_splits (int): Number of splits for reconstruction mode. Defaults to 1.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing predictions and metadata. Keys include:
                - "meth": Original methylation tensor.
                - "mask_na": Mask for missing values.
                - Additional keys produced by the forward pass.

        """
        # Step 1: Preprocess Data
        meth, mask_na = self._get_meth_and_na_mask(batch)
        safe_clone(meth)

        # Step 2: Initialize Input Masks
        current_input_masks = self._initialize_input_masks(meth, mask_na, n_splits)

        # Step 3: Prepare Input Data
        input_data = self._pad_input_data(
            batch,
            meth,
            mask_na,
            current_input_masks,
            binarize_input=self.hparams.training.get("binarize_input", False),
        )

        # Step 4: Forward Pass
        full_sequence_embeddings = self.net.encode_sequence(batch["dna_embeddings"])
        output_dictionary = {
            "meth": meth,
            "mask_na": mask_na,
            "current_input_masks": current_input_masks,
            "chroms": batch["chroms"],
            "positions": batch["positions"],
        }
        return self._forward_pass(
            batch,
            input_data,
            output_dictionary,
            full_sequence_embeddings,
            current_input_masks,
        )

    def _predict_attention_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Perform a prediction step focusing on attention outputs.

        This method uses a forward pass and a hook to extract attention weights from a
        specified layer.

        Args:
            batch (Dict[str, Any]): Input batch containing data for prediction.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - "attention_weights": The extracted attention weights tensor.

        """
        layer_index = getattr(self, "layer_index_predict", -1)
        aggregate_heads = getattr(self, "aggregate_heads_predict", "mean")

        meth, mask_na = self._get_meth_and_na_mask(batch)
        current_input_masks = self._initialize_input_masks(meth, mask_na)
        input_data = self._pad_input_data(
            batch,
            meth,
            mask_na,
            current_input_masks,
            binarize_input=self.hparams.training.get("binarize_input", False),
        )

        save_output = SaveOutput()
        target_layer = self.net.transformer_encoder.layers[layer_index].self_attn

        with patch_attention(target_layer):
            hook_handle = target_layer.register_forward_hook(save_output)
            _ = self.net.encode_sample(**input_data)
            hook_handle.remove()

        attention_weights = save_output.outputs[0]

        if aggregate_heads == "mean":
            attention_weights = attention_weights.mean(dim=1)
        elif aggregate_heads == "max":
            attention_weights = attention_weights.max(dim=1)[0]
        elif aggregate_heads != "none":
            msg = "aggregate_heads must be 'mean', 'max' or 'none'"
            raise ValueError(msg)

        # Clear any saved outputs from GPU memory
        save_output.clear()

        return {
            "attention_weights": attention_weights,
            "meth": input_data["meth"],
            "mask_na": input_data["mask_na"],
            "current_input_masks": current_input_masks,
            "chroms": input_data["chroms"],
            "positions": input_data["positions"],
        }

    def _predict_reconstruct_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Perform a reconstruction prediction step using pre-loaded embeddings.

        Args:
            batch (Dict[str, Any]): Input batch for prediction.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing reconstructed methylation values
                and metadata.

        """
        n_thinking_steps = getattr(self, "n_thinking_steps_predict", 0)
        thinking_step_size = getattr(self, "thinking_step_size_predict", 500)
        uncertainty_quantile = getattr(self, "uncertainty_quantile_predict", 0.1)

        if not hasattr(self, "cached_embeddings"):
            msg = "Embeddings not initialized. Ensure on_predict_epoch_start was called."
            raise ValueError(
                msg,
            )

        if not hasattr(self, "all_cached_embeddings") and n_thinking_steps > 0:
            msg = "Embeddings not initialized. Ensure on_predict_epoch_start was called."
            raise ValueError(
                msg,
            )

        return self._chain_of_thought_forward_pass(
            batch,
            n_thinking_steps,
            thinking_step_size,
            uncertainty_quantile,
            self.cached_embeddings,
            n_splits=1,
        )

    def _get_meth_and_na_mask(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Handles missing values and returns the methylation tensor and the mask for na values.

        Args:
            batch (Dict[str, Any]): A dictionary containing batch data

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Methylation tensor and mask for missing values

        """
        meth = batch["meth"]
        mask_na = torch.isnan(meth)
        meth = torch.nan_to_num(meth, nan=0.0)
        return meth, mask_na

    def _initialize_input_masks(
        self,
        meth: torch.Tensor,
        mask_na: torch.Tensor,
        n_splits: int = 1,
    ) -> torch.Tensor:
        """Initialize input masks by randomly selecting sites for each sample.

        Args:
            meth (torch.Tensor): Methylation tensor (batch_size, n_sites)
            mask_na (torch.Tensor): Missing values mask (batch_size, n_sites)
            n_splits (int): Number of splits to divide sites into. Defaults to 1.

        Returns:
            torch.Tensor: Boolean tensor (batch_size, n_sites) indicating selected input sites

        """
        batch_size, n_sites = meth.shape
        sites_per_split = n_sites // n_splits
        device = meth.device

        input_masks = torch.zeros((batch_size, n_sites), dtype=torch.bool, device=device)
        for i in range(batch_size):
            perm = torch.randperm(n_sites, device=device)[:sites_per_split]
            input_masks[i, perm] = True
        input_masks[mask_na] = False

        return input_masks

    def _pad_input_data(
        self,
        batch: dict[str, Any],
        meth: torch.Tensor,
        mask_na: torch.Tensor,
        current_input_masks: torch.Tensor,
        binarize_input: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Prepares input data with padding to handle variable-length sequences.

        Args:
            batch (Dict[str, Any]): Input batch containing sequence embeddings and positions
            meth (torch.Tensor): Methylation tensor
            mask_na (torch.Tensor): Mask of missing values
            current_input_masks (torch.Tensor): Boolean tensor indicating selected input
                sites per sample
            binarize_input (bool, optional): Whether to binarize the methylation input.
                Defaults to False.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing input data tensors with padding

        """
        batch_size = meth.size(0)
        inputs = ["meth", "sequence_embeddings", "chroms", "positions", "mask_na"]
        input_tensors = {key: [] for key in inputs}

        sequence_embeddings = self.net.encode_sequence(batch["dna_embeddings"])

        for i in range(batch_size):
            mask = current_input_masks[i]
            input_tensors["meth"].append(meth[i, mask])
            input_tensors["sequence_embeddings"].append(sequence_embeddings[i, mask])
            input_tensors["chroms"].append(batch["chroms"][i, mask])
            input_tensors["positions"].append(batch["positions"][i, mask])
            input_tensors["mask_na"].append(mask_na[i, mask])

        padding_values = {
            "meth": 0.0,
            "sequence_embeddings": 0.0,
            "chroms": -1,
            "positions": -1,
            "mask_na": True,
        }

        for key in inputs:
            input_tensors[key] = torch.nn.utils.rnn.pad_sequence(
                input_tensors[key],
                batch_first=True,
                padding_value=padding_values[key],
            )

        if binarize_input:
            input_tensors["meth"] = torch.bernoulli(input_tensors["meth"])

        return input_tensors

    def _forward_pass(
        self,
        batch: dict[str, Any],
        input_data: dict[str, torch.Tensor],
        loss_inputs: dict[str, Any],
        full_sequence_embeddings: torch.Tensor,
        current_input_masks: torch.Tensor,
    ) -> dict[str, Any]:
        """Performs the forward pass and updates loss inputs.

        Args:
            batch (Dict[str, Any]): Batch dictionary
            input_data (Dict[str, torch.Tensor]): Input data dictionary
            loss_inputs (Dict[str, Any]): Dictionary of inputs for loss calculation
            full_sequence_embeddings (torch.Tensor): Full sequence embeddings
            current_input_masks (torch.Tensor): Boolean tensor indicating selected input
                sites per sample

        Returns:
            Dict[str, Any]: Dictionary of loss inputs

        """
        # Encode sample
        sample_embedding = self.net.encode_sample(**input_data)

        loss_inputs["sample_embedding"] = sample_embedding

        # Add methylation predictions
        pred_meth, pred_meth_unc = self.net.query_methylation(
            sample_embedding,
            full_sequence_embeddings,
            m_or_beta="m",
        )
        loss_inputs.update(
            {
                "pred_meth": pred_meth,
                "pred_meth_unc": pred_meth_unc,
            },
        )

        # Add condition predictions if enabled
        if self.net.use_condition_decoder:
            loss_inputs.update(
                {
                    "pred_conditions": self.net.query_condition(sample_embedding),
                    "obsm": batch["obsm"],
                },
            )

        if self.hparams.training["reconstruct_mode"] == "unseen" or (
            not self.training and self.hparams.training["generative_splits"] != 1
        ):
            loss_inputs["loss_mask"] = ~current_input_masks

        return loss_inputs

    def _chain_of_thought_forward_pass(
        self,
        batch: dict[str, Any],
        n_thinking_steps: int,
        thinking_step_size: int,
        uncertainty_quantile: float,
        embeddings_to_predict: torch.Tensor,
        n_splits: int = 1,
    ) -> dict[str, torch.Tensor]:
        """Unrolls a 'chain-of-thought' reconstruction logic during training.

        Each iteration:
        1) Performs a forward pass with partial or updated inputs.
        2) Measures uncertainties and picks the next batch of sites to 'reveal'.
        3) Updates the working_batch for the next iteration.

        Returns:
            The final iteration's outputs (e.g., pred_meth, pred_meth_unc, etc.).
            The 'training_step' can then compute losses on these final predictions.

        """
        # Step 1: Make a local copy of the batch (to avoid overwriting the original)
        working_batch = dict(batch.items())

        # For tensor values that need to be modified, create deep copies
        for k in ["meth", "dna_embeddings"]:
            if k in working_batch and torch.is_tensor(working_batch[k]):
                working_batch[k] = safe_clone(working_batch[k])

        # Step 2: Pre-compute all sequence embeddings for thinking steps if needed
        if n_thinking_steps > 0 and hasattr(self, "all_cached_embeddings"):
            # Pre-select random indices for all thinking steps at once
            all_subset_indices = []
            # Calculate how many sites to preselect based on uncertainty_quantile
            preselect_size = int(thinking_step_size / uncertainty_quantile)
            preselect_size = min(preselect_size, self.all_cached_embeddings.shape[1])

            for _ in range(n_thinking_steps):
                subset_indices = torch.randperm(
                    self.all_cached_embeddings.shape[1], device=self.all_cached_embeddings.device
                )[:preselect_size]
                all_subset_indices.append(subset_indices)

        # Step 3: Loop over 'n_thinking_steps' to iteratively refine your predictions
        for i in range(n_thinking_steps + 1):
            # Step 3.1: Prepare the standard 'meth' and 'mask_na'
            meth, mask_na = self._get_meth_and_na_mask(working_batch)

            # Step 3.2: Create an input mask for which sites to feed in this iteration
            current_input_masks = self._initialize_input_masks(meth, mask_na, n_splits=n_splits)

            # Step 3.3: Pad the input data (like in your existing code)
            input_data = self._pad_input_data(
                working_batch,
                meth,
                mask_na,
                current_input_masks,
                binarize_input=self.hparams.training.get("binarize_input", False),
            )

            # Step 3.4: Decide which DNA embeddings to pass into the net
            if i < n_thinking_steps:
                subset_indices = all_subset_indices[i]
                full_sequence_embeddings = self.net.encode_sequence(
                    self.all_cached_embeddings[:, subset_indices, :]
                )
            else:
                # Final step uses the actual embeddings to predict
                full_sequence_embeddings = self.net.encode_sequence(embeddings_to_predict)

            # Step 3.5: Expand across batch dimension if needed
            if full_sequence_embeddings.dim() == 2:
                full_sequence_embeddings = full_sequence_embeddings.unsqueeze(0).expand(meth.size(0), -1, -1)
            elif full_sequence_embeddings.size(0) == 1 and meth.size(0) > 1:
                full_sequence_embeddings = full_sequence_embeddings.expand(meth.size(0), -1, -1)

            # Step 3.6: Forward pass with your net, capturing predictions & uncertainties
            step_outputs = {
                "meth": safe_clone(meth),
                "mask_na": safe_clone(mask_na),
            }
            step_outputs = self._forward_pass(
                working_batch,  # entire batch dictionary
                input_data,  # masked/padded input
                step_outputs,  # partial dictionary we pass in
                full_sequence_embeddings,  # partial or full embeddings
                current_input_masks,
            )

            # Step 3.7: If this is not the final iteration, refine 'working_batch'
            #      with the predictions we consider reliable (lowest uncertainty)
            if i < n_thinking_steps:
                pred_meth = safe_clone(step_outputs["pred_meth"])

                # Step 3.7.1: Pick the top fraction or quantile of confident predictions
                uncertainty_values = safe_clone(step_outputs["pred_meth_unc"])
                threshold = torch.quantile(
                    uncertainty_values, uncertainty_quantile, dim=1, keepdim=True
                )
                confident_mask = uncertainty_values <= threshold
                confident_indices = subset_indices[
                    safe_clone(confident_mask[0])
                    .detach()
                    .to(device=subset_indices.device, dtype=torch.bool)
                ]

                # Step 3.7.2: Update the working batch with the confident predictions
                working_batch["meth"] = torch.cat(
                    (
                        safe_clone(working_batch["meth"]),
                        m_to_beta(pred_meth[:, confident_mask[0]]),
                    ),
                    dim=1,
                )
                working_batch["dna_embeddings"] = torch.cat(
                    (
                        safe_clone(working_batch["dna_embeddings"]),
                        self.all_cached_embeddings[:, confident_indices, :].expand(
                            meth.size(0), -1, -1
                        ),
                    ),
                    dim=1,
                )
                working_batch["chroms"] = torch.cat(
                    (
                        safe_clone(working_batch["chroms"]),
                        self.chromosomes_predict[:, confident_indices].expand(meth.size(0), -1),
                    ),
                    dim=1,
                )
                working_batch["positions"] = torch.cat(
                    (
                        safe_clone(working_batch["positions"]),
                        self.positions_predict[:, confident_indices].expand(meth.size(0), -1),
                    ),
                    dim=1,
                )

                # Step 3.7.3: Sort the working batch
                chroms = working_batch["chroms"].detach().cpu().numpy()[0]  # shape = (num_sites,)
                positions = working_batch["positions"].detach().cpu().numpy()[0]
                working_batch = self._apply_sorting_strategy(
                    chroms,
                    positions,
                    working_batch,
                    self.trainer.datamodule.hparams["sorting_strategy"],
                )

        # Step 4: Return the final iteration's outputs
        step_outputs["meth"] = safe_clone(input_data["meth"])
        step_outputs["mask_na"] = safe_clone(input_data["mask_na"])

        return step_outputs

    def _apply_sorting_strategy(
        self,
        chroms: np.ndarray,
        positions: np.ndarray,
        batch: dict[str, torch.Tensor],
        sorting_strategy: str,
    ) -> dict[str, torch.Tensor]:
        """Apply the specified sorting strategy to the data.

        Sorts CpG sites based on the chosen strategy:
        - sorted_chromosome: Sort by position within each chromosome
        - random_chromosome: Random order within each chromosome
        - random: Completely random order
        - original: Keep original order

        Args:
            chroms (np.ndarray): Array of chromosome indices
            positions (np.ndarray): Array of genomic positions
            batch (tuple[torch.Tensor, ...]): Tuple of tensors containing batch data
            sorting_strategy (str): Strategy to use for sorting the data

        Returns:
            tuple[torch.Tensor, ...]: Sorted batch tensors according to the chosen strategy

        """
        if "chromosome" in sorting_strategy:
            unique_chroms, chrom_indices = np.unique(chroms, return_inverse=True)

            sorted_indices = []
            for chrom in unique_chroms:
                chrom_mask = chrom_indices == np.where(unique_chroms == chrom)[0][0]
                chrom_positions = positions[chrom_mask]
                if "sorted" in sorting_strategy:
                    sorted_chrom_indices = np.argsort(chrom_positions)
                elif "random" in sorting_strategy:
                    sorted_chrom_indices = self.rng.permutation(len(chrom_positions))
                sorted_indices.extend(np.where(chrom_mask)[0][sorted_chrom_indices])

            # Shuffle chromosomes
            shuffled_chroms = self.rng.permutation(unique_chroms)

            # Apply the shuffle order to sorted indices
            final_sorted_indices = []
            for chrom in shuffled_chroms:
                chrom_mask = chroms == chrom
                chrom_sorted_indices = [i for i in sorted_indices if chrom_mask[i]]
                final_sorted_indices.extend(chrom_sorted_indices)

            batch = {
                k: (
                    tensor[:, final_sorted_indices]
                    if tensor.dim() == 2
                    else tensor[:, final_sorted_indices, :]
                    if tensor.dim() == 3
                    else tensor
                )
                if tensor is not None
                else None
                for k, tensor in batch.items()
            }

        elif sorting_strategy == "random":
            random_indices = self.rng.permutation(len(positions))
            batch = {k: tensor[random_indices] for k, tensor in batch.items()}

        return batch

    def _update_input_masks_and_meth(
        self,
        meth: torch.Tensor,
        meth_original: torch.Tensor,
        mask_na: torch.Tensor,
        loss_inputs: dict[str, Any],
        current_input_masks: torch.Tensor,
        previous_input_masks: torch.Tensor,
        n_splits: int,
        total_steps: int,
        current_step: int,
        epsilon: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the input sites per sample and the meth tensor based on weighted error.

        Args:
            meth (torch.Tensor): Current methylation tensor
            meth_original (torch.Tensor): Original methylation tensor
            mask_na (torch.Tensor): Mask of missing values
            loss_inputs (Dict[str, Any]): Dictionary containing predictions
            current_input_masks (torch.Tensor): Current input masks tensor
            previous_input_masks (torch.Tensor): Previous input masks tensor
            n_splits (int): Number of splits
            total_steps (int): Total training steps
            current_step (int): Current global step
            epsilon (float, optional): Small constant to avoid division by zero. Defaults to 1e-6.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Updated methylation tensor
                - Updated input masks

        """
        pred_meth = loss_inputs["pred_meth"].detach()
        pred_meth_unc = loss_inputs["pred_meth_unc"].detach()
        batch_size, n_sites = meth.shape

        meth_error = F.l1_loss(pred_meth, beta_to_m(meth_original), reduction="none")
        meth_error_weighted = meth_error / (pred_meth_unc + epsilon)

        for i in range(batch_size):
            input_mask_i = current_input_masks[i]
            valid_site_mask = (~mask_na[i]) & (~input_mask_i)
            site_errors = safe_clone(meth_error_weighted[i])
            site_errors[~valid_site_mask] = float("-inf")

            # Calculate number of sites to unmask
            num_sites_to_add = min(valid_site_mask.sum(), n_sites // n_splits)
            if num_sites_to_add > 0:
                # Get top k errors and set those sites to be unmasked in next step's mask
                _, top_error_indices = torch.topk(site_errors, k=num_sites_to_add, largest=True)
                current_input_masks[i][top_error_indices] = True

                # Update meth with predicted values at newly unmasked sites
                newly_unmasked = current_input_masks[i] & (~previous_input_masks[i])
                update_prob = current_step / total_steps
                if torch.rand(1).item() < update_prob:
                    meth[i, newly_unmasked] = m_to_beta(pred_meth[i, newly_unmasked]).to(
                        meth.dtype,
                    )

        # Clone meth and current_input_masks to avoid errors with the computation graph
        return safe_clone(meth), safe_clone(current_input_masks)

    def _calculate_losses(
        self,
        **kwargs: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Calculate the losses for the model.

        Args:
            **kwargs: Dictionary containing any of the following:
                - pred_meth (torch.Tensor): Predicted methylation values
                - pred_meth_unc (torch.Tensor): Predicted uncertainty
                - meth (torch.Tensor): Target values
                - mask_na (torch.Tensor): Mask for NA values
                - sample_embedding (torch.Tensor): Sample embeddings
                - sample_embedding_full (torch.Tensor): Sample embeddings with full data
                - obsm (torch.Tensor): Observation matrix
                - pred_conditions (torch.Tensor): Predicted conditions
                - loss_mask (torch.Tensor): Mask for loss calculation

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing calculated losses

        """
        losses = {}

        # Get valid mask
        mask_na = kwargs.get("mask_na")
        meth = kwargs.get("meth")
        loss_mask = kwargs.get("loss_mask")

        if mask_na is not None:
            valid_mask = ~mask_na
        elif meth is not None:
            valid_mask = torch.ones_like(meth, dtype=bool)

        if loss_mask is not None:
            valid_mask = valid_mask & loss_mask

        # Calculate methylation losses if inputs available
        pred_meth = kwargs.get("pred_meth")
        pred_meth_unc = kwargs.get("pred_meth_unc")
        if pred_meth is not None and meth is not None:
            losses["m_mae"] = F.l1_loss(pred_meth[valid_mask], beta_to_m(meth[valid_mask])).mean()
            losses["m_mae_unc"] = F.l1_loss(
                F.l1_loss(pred_meth[valid_mask], beta_to_m(meth[valid_mask]), reduction="none"),
                pred_meth_unc[valid_mask],
            ).mean()

            # Beta-based losses using converted values
            losses["betas_mae"] = F.l1_loss(
                m_to_beta(pred_meth[valid_mask]),
                meth[valid_mask],
            ).mean()
            losses["betas_kld"] = kld_bernoulli_loss(
                m_to_beta(pred_meth[valid_mask]),
                meth[valid_mask],
            ).mean()
            losses["betas_beta"] = beta_loss(
                m_to_beta(pred_meth[valid_mask]),
                meth[valid_mask],
            ).mean()
            losses["betas_wd"] = wd_loss(m_to_beta(pred_meth[valid_mask]), meth[valid_mask]).mean()

        # Calculate embedding losses if available
        sample_embedding = kwargs.get("sample_embedding")
        if sample_embedding is not None:
            losses["contrastive"] = contrastive_loss(
                sample_embedding,
                self.hparams.training["contrastive_threshold"],
            )
            losses["sample_kld"] = kld_normal_loss(sample_embedding).mean()

        # Calculate consistency loss if available
        sample_embedding_full = kwargs.get("sample_embedding_full")
        if sample_embedding_full is not None:
            losses["consistency"] = consistency_loss(
                sample_embedding,
                sample_embedding_full,
            )

        # Calculate condition losses if available
        pred_conditions = kwargs.get("pred_conditions")
        obsm = kwargs.get("obsm")
        if pred_conditions is not None and obsm is not None:
            condition_loss_type = self.hparams.training.get("condition_decoder_loss")
            if condition_loss_type == "mae":
                losses["condition_loss"] = F.l1_loss(pred_conditions, obsm).mean()
            elif condition_loss_type == "mse":
                losses["condition_loss"] = F.mse_loss(pred_conditions, obsm).mean()
            elif condition_loss_type == "huber":
                losses["condition_loss"] = F.huber_loss(pred_conditions, obsm).mean()
            elif condition_loss_type == "cosine_similarity":
                losses["condition_loss"] = 1 - F.cosine_similarity(pred_conditions, obsm).mean()
            elif condition_loss_type == "ce":
                losses["condition_loss"] = F.cross_entropy(pred_conditions, obsm).mean()
            elif condition_loss_type == "bce":
                losses["condition_loss"] = F.binary_cross_entropy_with_logits(
                    pred_conditions,
                    obsm,
                ).mean()

            # Check for NaN and replace with zero if found
            if "condition_loss" in losses and torch.isnan(losses["condition_loss"]):
                losses["condition_loss"] = torch.tensor(
                    0.0, device=losses["condition_loss"].device
                )

        # Define which losses to include for validation
        validation_losses = [
            "m_mae",
            "m_mae_unc",
            "condition_loss",
        ]

        # Calculate the total loss with respective weights
        current_losses = losses.keys() if self.training else validation_losses
        total_loss = sum(
            losses[k] * self.hparams.training["loss_weights"].get(k, 0.0)
            for k in current_losses
            if k in losses
        )
        losses["loss"] = total_loss

        return losses

    def _log_training_metrics(
        self,
        losses: dict[str, torch.Tensor],
        batch_size: int,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Log training metrics.

        Args:
            losses: Dictionary of loss values
            batch_size: Size of the batch
            optimizer: The optimizer being used

        """
        self.train_loss(losses["loss"])
        self.log_dict(
            {f"train/{k}": v for k, v in losses.items()},
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        self.log(
            "trainer/lr",
            optimizer.param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "global_step",
            self.global_step,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )

if __name__ == "__main__":
    _ = CpGPTLitModule(None, None, None)
