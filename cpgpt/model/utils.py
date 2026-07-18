import contextlib
from typing import TYPE_CHECKING, Any

import torch
from lightning.pytorch.callbacks import Callback
from torch import nn

if TYPE_CHECKING:
    import pytorch_lightning as pl


class SaveDataModuleHyperparametersCallback(Callback):
    """A callback that saves data module hyperparameters to checkpoints.

    This callback ensures that the hyperparameters used for the data module are preserved
    when saving checkpoints, enabling reproducibility and model analysis.

    Args:
        hyper_parameters (dict): Dictionary containing the data module's hyperparameters.

    Attributes:
        hyper_parameters (dict): Stored hyperparameters to be saved with checkpoints.

    """

    def __init__(self, hyper_parameters: dict) -> None:
        """Initialize the callback with hyperparameters.

        Args:
            hyper_parameters (dict): Dictionary containing the data module's hyperparameters.

        """
        super().__init__()
        self.hyper_parameters = hyper_parameters

    def on_save_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        checkpoint: dict[str, Any],
    ) -> None:
        """Add hyperparameters to the checkpoint during saving.

        Args:
            trainer: The PyTorch Lightning trainer instance.
            pl_module: The current LightningModule being trained.
            checkpoint (dict): The checkpoint dictionary being saved.

        """
        checkpoint["datamodule_hyper_parameters"] = self.hyper_parameters


def beta_to_m(beta_values: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Convert methylation beta values to M-values.

    Applies the logit transformation: M = log2(beta / (1 - beta))
    Used in methylation analysis to convert between different value representations.

    Args:
        beta_values (torch.Tensor): Beta values in range [0, 1].
        epsilon (float, optional): Small value to prevent log(0). Defaults to 1e-6.

    Returns:
        torch.Tensor: M-values, same shape as input.

    """
    # Clip beta values to prevent numerical instability
    beta_values = torch.clamp(beta_values, min=epsilon, max=1 - epsilon)
    return torch.log2(beta_values / (1 - beta_values))


def m_to_beta(m_values: torch.Tensor) -> torch.Tensor:
    """Convert methylation M-values to beta values.

    Applies the inverse logit transformation: beta = 2^M / (2^M + 1)
    Used in methylation analysis to convert between different value representations.

    Args:
        m_values (torch.Tensor): M-values, typically in range [-inf, inf].

    Returns:
        torch.Tensor: Beta values clamped to [eps, 1-eps], same shape as input.

    """
    m_values = torch.clamp(m_values, min=-20, max=20)
    power_2_m = 2**m_values
    return power_2_m / (power_2_m + 1)


class SaveOutput:
    """A class to save and manage outputs from a module.

    This class is designed to be used as a hook in PyTorch modules to capture and store their
    outputs.
    """

    def __init__(self) -> None:
        """Initialize an empty list to store outputs."""
        self.outputs = []

    def __call__(
        self,
        module: nn.Module,
        module_in: torch.Tensor,
        module_out: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Capture and store the output of a module.

        Args:
            module (nn.Module): The module being processed.
            module_in (torch.Tensor): The input to the module.
            module_out (tuple[torch.Tensor, torch.Tensor]): The output from the module.

        """
        self.outputs.append(module_out[1])

    def clear(self) -> None:
        """Clear all stored outputs and ensure they are removed from memory."""
        self.outputs.clear()
        self.outputs = []


@contextlib.contextmanager
def patch_attention(m: nn.Module) -> None:
    """Temporarily patches an attention module to return attention weights.

    This context manager modifies the forward method of an attention module to always
    return attention weights, and restores the original behavior when exiting the context.

    Args:
        m (nn.Module): The attention module to be temporarily patched.

    Yields:
        None. The module is modified in-place within the context.

    Example:
        >>> with temporary_patch_attention(model.attention):
        ...     output = model(input_tensor)
        ...     # attention weights are available in output

    """
    # Save the original forward
    forward_orig = m.forward

    def wrap(
        *args: list[torch.Tensor],
        **kwargs: dict[str, bool | float | None],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Wrapper function for the original forward method.

        This wrapper ensures that attention weights are always computed and returned.

        Args:
            *args: Variable length list of tensors (query, key, value).
            **kwargs: Keyword arguments for the attention module.
                Supported keys include:
                - need_weights (bool): Whether to return attention weights
                - average_attn_weights (bool): Whether to average attention weights
                - key_padding_mask (Optional[Tensor]): Mask for padded elements
                - attn_mask (Optional[Tensor]): Mask to prevent attention to positions

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - output tensor of shape (batch_size, seq_len, hidden_size)
                - attention weights tensor of shape (batch_size, num_heads, seq_len, seq_len)

        """
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        return forward_orig(*args, **kwargs)

    # Patch
    m.forward = wrap
    try:
        yield
    finally:
        # Restore original
        m.forward = forward_orig


def safe_clone(tensor: torch.Tensor) -> torch.Tensor:
    """Safely clone a tensor only if it requires gradient.

    This function optimizes memory usage by only cloning tensors that require gradients.
    For tensors that don't require gradients, it returns the original tensor.

    Args:
        tensor (torch.Tensor): The input tensor to potentially clone.

    Returns:
        torch.Tensor: A cloned tensor if the input requires gradients,
            otherwise the original tensor.

    """
    return tensor.clone() if tensor.requires_grad else tensor
