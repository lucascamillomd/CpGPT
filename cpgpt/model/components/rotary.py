"""Rotary positional embeddings, vendored from torchtune 0.3.1.

Vendored verbatim (BSD-3-Clause, Meta Platforms) so that CpGPT does not depend on
torchtune/torchao, whose newer releases are incompatible with each other and with
torch>=2.13 (``torchao.dtypes.nf4tensor`` was removed). Numerics are identical to
the torchtune implementation the released checkpoints were trained with.
"""

import torch
from torch import nn


class RotaryPositionalEmbeddings(nn.Module):
    """Rotary positional embeddings (RoPE), https://arxiv.org/abs/2104.09864.

    Matches torchtune 0.3.1's implementation exactly.

    Args:
        dim (int): Embedding dimension per head (``embed_dim // num_heads``).
        max_seq_len (int): Maximum expected sequence length.
        base (int): Base for the geometric progression used to compute
            the rotation angles.

    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10_000) -> None:
        """Initialize the rotary embedding cache."""
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self) -> None:
        """Compute rotation frequencies and build the cos/sin cache."""
        theta = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        """Cache cos/sin rotation terms for positions up to max_seq_len."""
        seq_idx = torch.arange(max_seq_len, dtype=self.theta.dtype, device=self.theta.device)
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor, *, input_pos: torch.Tensor | None = None) -> torch.Tensor:
        """Apply rotary embeddings.

        Args:
            x (torch.Tensor): Input of shape (batch, seq_len, num_heads, head_dim).
            input_pos (torch.Tensor | None): Optional position indices of shape
                (batch, seq_len). Defaults to positions 0..seq_len-1.

        Returns:
            torch.Tensor: Rotated tensor with the same shape and dtype as the input.

        """
        seq_len = x.size(1)
        rope_cache = self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        x_out = x_out.flatten(3)
        return x_out.type_as(x)
