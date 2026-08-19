import warnings

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.core.datamodule import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader

from cpgpt.model.components.legacy_numerics import enable_legacy_numerics

DEFAULT_PRECISION = "16-mixed"


class CpGPTTrainer(Trainer):
    """Custom trainer class for CpGPT model.

    Extends PyTorch Lightning's Trainer with custom prediction functionality.

    Precision defaults to ``"16-mixed"``: the released CpGPT checkpoints were trained
    and validated under 16-mixed autocast, and (together with the legacy-numerics
    shim) this keeps predictions bit-identical to the published torch-2.6 CUDA
    outputs. Overriding it changes predictions and is warned against.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the trainer with RichProgressBar callback if progress bar is enabled."""
        # Reproduce the torch<=2.6 RMSNorm numerics the checkpoints were trained with.
        enable_legacy_numerics()

        if "precision" not in kwargs:
            kwargs["precision"] = DEFAULT_PRECISION
        elif str(kwargs["precision"]) != DEFAULT_PRECISION:
            warnings.warn(
                f"CpGPTTrainer precision={kwargs['precision']!r} differs from the "
                f"validated default {DEFAULT_PRECISION!r}. CpGPT checkpoints were "
                "trained and evaluated under 16-mixed autocast; other settings "
                "produce predictions that are NOT reproducible against published "
                "CpGPT results.",
                UserWarning,
                stacklevel=2,
            )

        # Only add RichProgressBar if progress bars are enabled (default is True)
        enable_progress_bar = kwargs.get("enable_progress_bar", True)

        if enable_progress_bar:
            # Add RichProgressBar to callbacks
            callbacks = kwargs.get("callbacks", [])
            if not isinstance(callbacks, list):
                callbacks = [callbacks]

            # Add RichProgressBar if not already present
            if not any(isinstance(callback, RichProgressBar) for callback in callbacks):
                callbacks.append(RichProgressBar())

            kwargs["callbacks"] = callbacks

        super().__init__(**kwargs)

    def _concat_padded_tensors(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        """Concatenates either 2D or 3D tensors with padding on the remaining dimensions.

        Concatenation is done along dim=0 (batch dimension).

        Supported shapes:
          - 2D: (batch_size, seq_len)
          - 3D: (batch_size, seq_len, seq_len) or (batch_size, dim1, dim2)

        """
        if not tensors:
            return torch.tensor([])

        # Check number of dims to decide how we pad
        dims = len(tensors[0].shape)
        if dims not in {2, 3}:
            msg = f"_concat_padded_tensors only supports 2D or 3D, got {dims}D."
            raise ValueError(msg)

        if dims == 2:
            # Example shape: (B, L)
            max_len = max(t.size(1) for t in tensors)
            padded_tensors = [
                torch.nn.functional.pad(
                    t,
                    (0, max_len - t.size(1)),
                    value=float("nan") if torch.is_floating_point(t) else -1,
                )  # pad last dimension
                for t in tensors
            ]
            return torch.cat(padded_tensors, dim=0)

        # Example shape: (B, L, L2)
        max_dim1 = max(t.size(1) for t in tensors)
        max_dim2 = max(t.size(2) for t in tensors)
        # F.pad padding format for 2D inner: (left, right, top, bottom)
        # so we pad dimension 2 with (0, max_dim2 - t.size(2))
        # then dimension 1 with (0, max_dim1 - t.size(1))
        padded_tensors = [
            torch.nn.functional.pad(
                t,
                (
                    0,
                    max_dim2 - t.size(2),  # pad along last dim
                    0,
                    max_dim1 - t.size(1),  # pad along second-last dim
                ),
                value=float("nan") if torch.is_floating_point(t) else -1,
            )
            for t in tensors
        ]
        return torch.cat(padded_tensors, dim=0)

    def predict(
        self,
        model: torch.nn.Module,
        dataloaders: DataLoader | list[DataLoader] | None = None,
        datamodule: LightningDataModule | None = None,
        **kwargs: str | float | bool | Tensor,
    ) -> dict[str, Tensor]:
        """Runs prediction on the given model and dataloaders.

        This method:
          1) Injects arbitrary **kwargs (e.g., "predict_mode", "layer_index", "species", etc.)
             as attributes on the model (with a "_predict" suffix) for use in the model's
             predict_step.
          2) Calls the parent class's .predict method.
          3) Removes these temporary attributes from the model.
          4) Concatenates the resulting predictions from each batch (and each dataloader if
             multiple are used) into a dictionary of tensors.

        The CpGPTLitModule supports multiple "predict_mode" options:
        • "forward" (default) - Performs the standard forward pass, returning predicted methylation
          values, optional condition predictions, and other outputs.
        • "reconstruct" - Reconstructs or fills missing methylation values given genomic locations.
        • "attention" - Extracts attention weights from a specified transformer layer for
          inspection.

        Examples:
            1) Forward mode:
               trainer.predict(
                   model,
                   dataloaders,
                   predict_mode="forward",
                   return_keys=["sample_embeddings", "mask_na"],
               )

            2) Reconstruction mode:
               trainer.predict(
                   model,
                   dataloaders,
                   predict_mode="reconstruct",
                   species="homo_sapiens",
                   n_thinking_steps=0,
                   thinking_step_size=500,
                   uncertainty_quantile=0.1,
                   genomic_locations=["12:10000", "X:10100"],
                   return_keys=["pred_meth", "pred_meth_unc"],
               )

            3) Attention mode:
               trainer.predict(
                   model,
                   dataloaders,
                   predict_mode="attention",
                   layer_index=-1,
                   aggregate_heads="mean",
                   return_keys=["attention_weights", "chroms", "positions"],
               )

        Args:
            model (torch.nn.Module): The CpGPTLitModule model.
            dataloaders (DataLoader | list[DataLoader] | None, optional): DataLoader(s) containing
                the prediction data. Defaults to None.
            datamodule (LightningDataModule | None, optional): LightningDataModule for loading
                prediction data. Defaults to None.
            **kwargs (str | float | bool | Tensor): Additional arguments for the model's
                predict_step method. Relevant options include:
                • predict_mode (str): One of "forward", "attention", or "reconstruct".
                • return_keys (list[str] | None): Which prediction keys to include in the output
                  dictionary. Defaults to None (all keys).
                • layer_index (int): Which layer's attention to extract (for attention mode).
                • aggregate_heads (str): How to aggregate attention heads, e.g. "mean", "max",
                  "none" (for attention mode).
                • species (str): Genome assembly (for reconstruct mode).
                • genomic_locations (list[str]): Locations to reconstruct (for reconstruct mode).
                • n_thinking_steps (int): Number of thinking steps (for reconstruct mode). Defaults
                  to 0.
                • thinking_step_size (int): Size of each thinking step (for reconstruct mode).
                  Defaults to 500.
                • uncertainty_quantile (float): Uncertainty quantile (for reconstruct mode).
                  Defaults to 0.1.

        Returns:
            dict[str, torch.Tensor]: A dictionary where each key is a concatenated tensor
            corresponding to one type of output across all predicted batches (and dataloaders).
            Any missing keys in intermediate batches are omitted from the final dictionary.

        """
        # Pass all kwargs to the model's predict_step
        for key, value in kwargs.items():
            setattr(model, f"{key}_predict", value)

        # Only pass the arguments that the parent predict method expects
        predictions_list = super().predict(model, dataloaders, datamodule)

        # Clean up the attributes we added
        for key in kwargs:
            delattr(model, f"{key}_predict")

        # If predictions_list is empty, return empty dict
        if not predictions_list:
            return {}

        # Get all keys from the first prediction
        first_pred_keys = predictions_list[0].keys()

        # Create concatenated outputs for each key where possible
        return {
            key: self._concat_padded_tensors(
                [p[key] for p in predictions_list if key in p and p[key] is not None]
            )
            for key in first_pred_keys
            if any(key in p and p[key] is not None for p in predictions_list)
        }
