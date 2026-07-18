import os
import sys
from typing import Any

import numpy as np
import pandas as pd
import schedulefree
import torch
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler

from cpgpt.loss.loss import (
    c_index_loss,
    censored_mae_loss,
    cph_loss,
    gompertz_aft_loss,
    rsf_loss,
)

from .cpgpt_module import CpGPTLitModule
from .utils import (
    safe_clone,
)


class CpGPTMortalityLitModule(CpGPTLitModule):
    """A LightningModule for CpG methylation-based mortality prediction.

    Extends the base CpGPTLitModule with specific functionality for survival analysis and
    mortality prediction using methylation data.

    Args:
        Same as CpGPTLitModule

    """

    def __init__(
        self,
        training: dict[str, Any],
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        compile: bool = False,
    ) -> None:
        """Initialize the CpGPTMortalityLitModule."""
        super().__init__(training, net, optimizer, scheduler, compile)

        # Initialize prediction storage
        self.train_predictions = []
        self.train_times = []
        self.train_events = []
        self.train_ages = []
        self.train_females = []
        self.val_predictions = []
        self.val_times = []
        self.val_events = []
        self.val_ages = []
        self.val_females = []

    def on_train_epoch_start(self) -> None:
        """Initialize lists to store predictions and targets at the start of training epoch."""
        self.train_predictions = []
        self.train_times = []
        self.train_events = []
        self.train_ages = []
        self.train_females = []

    def on_validation_epoch_start(self) -> None:
        """Initialize lists to store predictions and targets at the start of validation epoch."""
        self.val_predictions = []
        self.val_times = []
        self.val_events = []
        self.val_ages = []
        self.val_females = []

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Execute a single training step with additional mortality prediction tracking."""
        # Call the parent class implementation
        loss = super().training_step(batch, batch_idx)

        # Extract time, event and predictions
        if "obsm" in batch and batch["obsm"] is not None and batch["obsm"].size(1) >= 2:
            time = batch["obsm"][:, 0]
            event = batch["obsm"][:, 1]
            age = batch["obsm"][:, 2]
            female = batch["obsm"][:, 3]
            # Get the last forward pass outputs, which should contain pred_conditions
            # This requires modifying the main training_step to store its final loss_inputs
            if hasattr(self, "_last_loss_inputs") and "pred_conditions" in self._last_loss_inputs:
                pred_risk = self._last_loss_inputs["pred_conditions"].squeeze().detach()
                pred_risk = torch.exp(pred_risk) * torch.exp(0.1 * pred_risk)

                self.train_predictions.append(pred_risk)
                self.train_times.append(time)
                self.train_events.append(event)
                self.train_ages.append(age)
                self.train_females.append(female)

        return loss

    def validation_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> None:
        """Perform a single validation step with additional mortality metrics collection."""
        # Run the parent validation step first
        super().validation_step(batch, batch_idx)

        # Get predictions and targets for c-index calculation
        if "obsm" in batch and batch["obsm"] is not None and batch["obsm"].size(1) >= 2:
            time = batch["obsm"][:, 0]
            event = batch["obsm"][:, 1]
            age = batch["obsm"][:, 2]
            female = batch["obsm"][:, 3]

            # This relies on the base class's _shared_val_step having been called
            # and the predictions being available
            if hasattr(self, "_last_val_outputs") and "pred_conditions" in self._last_val_outputs:
                pred_conditions = self._last_val_outputs["pred_conditions"]

                # Handle different prediction formats based on loss function
                if self.hparams.training.get("condition_decoder_loss") == "weibull_nll_loss":
                    shape = pred_conditions[:, 0]
                    scale = pred_conditions[:, 1]
                    remaining_life = scale * torch.exp(torch.lgamma(1 + 1 / shape))
                    age = batch["obsm"][:, 2]
                    total_risk = 1 / (remaining_life + age)
                    pred_risk = -total_risk
                elif self.hparams.training.get("condition_decoder_loss") == "gompertz_nll_loss":
                    lambda_param = pred_conditions[:, 0]
                    gamma = pred_conditions[:, 1]
                    age = batch["obsm"][:, 2]
                    risk = lambda_param * torch.exp(gamma * age)
                    pred_risk = -risk
                else:
                    pred_risk = pred_conditions.squeeze()

                if pred_risk.numel() > 0 and time.numel() > 0 and event.numel() > 0:
                    pred_risk = pred_risk.detach().cpu()
                    pred_risk = torch.exp(pred_risk) * torch.exp(0.1 * pred_risk)
                    time = time.detach().cpu()
                    event = event.detach().cpu()

                    # Ensure tensors are at least one-dimensional
                    if pred_risk.dim() == 0:
                        pred_risk = pred_risk.unsqueeze(0)
                    if time.dim() == 0:
                        time = time.unsqueeze(0)
                    if event.dim() == 0:
                        event = event.unsqueeze(0)

                    self.val_predictions.append(pred_risk)
                    self.val_times.append(time)
                    self.val_events.append(event)
                    self.val_ages.append(age)
                    self.val_females.append(female)

    def on_train_epoch_end(self) -> None:
        """Calculate c-index at the end of training epoch."""
        # Calculate c-index if we have predictions
        if len(self.train_predictions) > 0:
            # Concatenate all predictions and targets
            all_predictions = torch.cat(self.train_predictions)
            all_times = torch.cat(self.train_times)
            all_events = torch.cat(self.train_events)
            all_ages = torch.cat(self.train_ages)
            all_females = torch.cat(self.train_females)

            # Convert to numpy and create DataFrame for CoxPH
            df = pd.DataFrame(
                {
                    "duration": all_times.cpu().numpy(),
                    "event": all_events.cpu().numpy(),
                    "predictions": all_predictions.cpu().numpy()
                    if all_predictions.ndim == 1
                    else all_predictions.cpu().numpy()[:, 0],
                    "age": all_ages.cpu().numpy(),
                    "female": all_females.cpu().numpy(),
                }
            )

            # Fit CoxPH model
            try:
                # Standardize predictions, age, and female with a single scaler
                scaler = StandardScaler()
                features_to_scale = ["predictions", "age", "female"]
                df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

                # Temporarily redirect stdout to suppress CoxPH output
                original_stdout = sys.stdout
                sys.stdout = open(os.devnull, "w")

                cph = CoxPHFitter()
                cph.fit(df, "duration", "event", ["predictions", "age", "female"], robust=True)

                # Restore stdout
                sys.stdout.close()
                sys.stdout = original_stdout

                # Calculate z-score manually from coefficient and standard error
                coef = cph.params_["predictions"]
                se = cph.standard_errors_["predictions"]
                z_score = abs(coef / se)

                # Calculate hazard ratio
                hazard_ratio = np.exp(coef)

                # Calculate c-index
                c_index = concordance_index(df["duration"], -df["predictions"], df["event"])

                # Log metrics
                self.log("train/c_index", c_index, sync_dist=True, prog_bar=True)
                self.log("train/z_score", z_score, sync_dist=True, prog_bar=True)
                self.log("train/hazard_ratio", hazard_ratio, sync_dist=True, prog_bar=True)

            except Exception:
                pass

    def on_validation_epoch_end(self) -> None:
        """Calculate c-index and track best values at the end of validation epoch."""
        # Call parent method to handle the base metrics
        super().on_validation_epoch_end()

        # Calculate c-index if we have predictions
        if len(self.val_predictions) > 0:
            # Concatenate all predictions and targets
            all_predictions = torch.cat(self.val_predictions)
            all_times = torch.cat(self.val_times)
            all_events = torch.cat(self.val_events)
            all_ages = torch.cat(self.val_ages)
            all_females = torch.cat(self.val_females)

            # Convert to numpy and create DataFrame for CoxPH
            df = pd.DataFrame(
                {
                    "duration": all_times.cpu().numpy(),
                    "event": all_events.cpu().numpy(),
                    "predictions": all_predictions.cpu().numpy()
                    if all_predictions.ndim == 1
                    else all_predictions.cpu().numpy()[:, 0],
                    "age": all_ages.cpu().numpy(),
                    "female": all_females.cpu().numpy(),
                }
            )

            try:
                # Standardize predictions, age, and female with a single scaler
                scaler = StandardScaler()
                features_to_scale = ["predictions", "age", "female"]
                df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

                # Temporarily redirect stdout to suppress CoxPH output
                original_stdout = sys.stdout
                sys.stdout = open(os.devnull, "w")

                # Fit CoxPH model
                cph = CoxPHFitter()
                cph.fit(df, "duration", "event", ["predictions", "age", "female"], robust=True)

                # Restore stdout
                sys.stdout.close()
                sys.stdout = original_stdout

                # Calculate z-score manually from coefficient and standard error
                coef = cph.params_["predictions"]
                se = cph.standard_errors_["predictions"]
                z_score = abs(coef / se)

                # Calculate hazard ratio
                hazard_ratio = np.exp(coef)

                # Calculate c-index
                c_index = concordance_index(df["duration"], -df["predictions"], df["event"])

                # Log metrics
                self.log("val/c_index", c_index, sync_dist=True, prog_bar=True)
                self.log("val/z_score", z_score, sync_dist=True, prog_bar=True)
                self.log("val/hazard_ratio", hazard_ratio, sync_dist=True, prog_bar=True)

            except Exception:
                # Ensure we use the same parameters for logging as in the try block
                self.log("val/c_index", 0.0, sync_dist=True, prog_bar=True)
                self.log("val/z_score", 0.0, sync_dist=True, prog_bar=True)
                self.log("val/hazard_ratio", 0.0, sync_dist=True, prog_bar=True)

    def _shared_val_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Perform a shared validation step and store outputs for mortality metrics."""
        # Create our own implementation that captures pred_conditions

        # Retrieve optimizer
        optimizer = self.optimizers()

        # If optimizer is schedulefree, set to eval mode
        if isinstance(optimizer, schedulefree.AdamWScheduleFree):
            optimizer.eval()

        # Step 1: Preprocess Data
        meth, mask_na = self._get_meth_and_na_mask(batch)
        meth_original = safe_clone(meth)

        # Step 2: Initialize Input Masks
        n_splits = 1
        current_input_masks = self._initialize_input_masks(meth, mask_na, n_splits=n_splits)

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

        # Save pred_conditions for validation metrics
        if "pred_conditions" in loss_inputs:
            self._last_val_outputs = {"pred_conditions": loss_inputs["pred_conditions"]}

        # Step 5: Calculate Losses
        return self._calculate_losses(**loss_inputs)

    def _forward_pass(
        self,
        batch: dict[str, Any],
        input_data: dict[str, torch.Tensor],
        loss_inputs: dict[str, Any],
        full_sequence_embeddings: torch.Tensor,
        current_input_masks: torch.Tensor,
    ) -> dict[str, Any]:
        """Performs the forward pass and saves outputs for mortality prediction."""
        # Call the parent implementation
        loss_inputs = super()._forward_pass(
            batch, input_data, loss_inputs, full_sequence_embeddings, current_input_masks
        )

        # Store the loss inputs for access in training_step
        self._last_loss_inputs = loss_inputs

        return loss_inputs

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
        # Filter obsm to match the number of columns in pred_conditions
        kwargs_filtered = kwargs.copy()
        pred_conditions = kwargs.get("pred_conditions")
        obsm = kwargs.get("obsm")
        if pred_conditions is not None and obsm is not None:
            if pred_conditions.shape[1] < obsm.shape[1]:
                kwargs_filtered["obsm"] = obsm[:, -pred_conditions.shape[1] :]

        # Call the parent implementation to get base losses with filtered kwargs
        losses = super()._calculate_losses(**kwargs_filtered)

        # Calculate condition losses if available
        pred_conditions = kwargs.get("pred_conditions")
        obsm = kwargs.get("obsm")
        if pred_conditions is not None and obsm is not None:
            condition_loss_type = self.hparams.training.get("condition_decoder_loss")
            if condition_loss_type == "censored_mae_loss":
                time, event = obsm[:, 0], obsm[:, 1]
                losses["condition_loss"] = censored_mae_loss(pred_conditions, time, event).mean()
            elif condition_loss_type == "c_index_loss":
                time, event, age, female = obsm[:, 0], obsm[:, 1], obsm[:, 2], obsm[:, 3]
                losses["condition_loss"] = c_index_loss(
                    pred_conditions, time, event, age, female
                ).mean()
            elif condition_loss_type == "cph_loss":
                time, event = obsm[:, 0], obsm[:, 1]
                losses["condition_loss"] = cph_loss(pred_conditions, time, event).mean()
            elif condition_loss_type == "gompertz_aft_loss":
                time, event, age, female = obsm[:, 0], obsm[:, 1], obsm[:, 2], obsm[:, 3]
                losses["condition_loss"] = gompertz_aft_loss(
                    pred_conditions,
                    time,
                    age,
                    female,
                    event,
                ).mean()
            elif condition_loss_type == "rsf_loss":
                time, event = obsm[:, 0], obsm[:, 1]
                losses["condition_loss"] = rsf_loss(pred_conditions, time, event).mean()

            # Check for NaN and replace with zero if found
            if "condition_loss" in losses and torch.isnan(losses["condition_loss"]):
                losses["condition_loss"] = torch.tensor(
                    0.0, device=losses["condition_loss"].device
                )

            losses["loss"] += (
                self.hparams.training["loss_weights"].get("condition_loss", 0.0)
                * losses["condition_loss"]
            )

        return losses


if __name__ == "__main__":
    _ = CpGPTMortalityLitModule(None, None, None)
