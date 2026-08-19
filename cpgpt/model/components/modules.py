import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Dropout, Linear, MultiheadAttention

from .rotary import RotaryPositionalEmbeddings


@torch.no_grad()
def create_hic_attention_mask(
    chroms: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    decay_factor: float = 1,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    """Create a genomic distance-based attention mask for Hi-C data.

    Generates an attention mask that incorporates both chromosomal and genomic distance
    information, similar to Hi-C contact matrices.

    Args:
        chroms (torch.Tensor): Chromosome indices of shape (batch_size, seq_length).
        positions (torch.Tensor): Genomic positions of shape (batch_size, seq_length).
        num_heads (int): Number of attention heads.
        decay_factor (float, optional): Power law decay factor. Defaults to 1.
        epsilon (float, optional): Small value to prevent numerical instability. Defaults to 1e-10.

    Returns:
        torch.Tensor: Attention mask of shape
            (batch_size * num_heads, seq_length + 1, seq_length + 1).

    """
    batch_size, seq_length = chroms.shape
    seq_length += 1  # Account for CLS token

    # Calculate genomic distances efficiently
    positions_expanded = positions.unsqueeze(2).expand(-1, -1, seq_length - 1)
    genomic_distance = (positions_expanded - positions_expanded.transpose(1, 2)).abs().float()

    # Create chromosome mask efficiently
    chroms_mask = (chroms.unsqueeze(2) == chroms.unsqueeze(1)).float()

    # Calculate distance bias
    distance_bias = (1.0 / (1.0 + genomic_distance)) ** decay_factor

    # Apply chromosome mask and cross-chromosome penalty
    bias = torch.zeros(
        (batch_size, seq_length, seq_length),
        device=positions.device,
        dtype=torch.float32,
    )
    bias[:, 1:, 1:] = chroms_mask * distance_bias

    # Clamp bias
    bias = torch.clamp(bias, min=epsilon)

    # Calculate attention mask from bias
    attn_mask = torch.log(bias)

    # Set CLS token attention
    attn_mask[:, 0, :] = 0.0  # CLS pays attention to all tokens
    attn_mask[:, :, 0] = 0.0  # All tokens pay attention to CLS

    slopes = get_alibi_slopes(num_heads).to(attn_mask.device)

    # Expand for multiple heads
    attn_mask = attn_mask.unsqueeze(1).expand(-1, num_heads, -1, -1) * slopes.view(
        1,
        num_heads,
        1,
        1,
    )
    return attn_mask.reshape(batch_size * num_heads, seq_length, seq_length)


def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """Calculate attention head slopes for ALiBi positional encoding.

    Implements the slope calculation for Attention with Linear Biases (ALiBi)
    as described in https://arxiv.org/abs/2108.12409.

    Args:
        n_heads (int): Number of attention heads.

    Returns:
        torch.Tensor: Slope values for each attention head.

    """
    n = 2 ** math.floor(math.log2(n_heads))
    m_0 = 2.0 ** (-8.0 / n)
    m = torch.pow(m_0, torch.arange(1, 1 + n))
    if n < n_heads:
        m_hat_0 = 2.0 ** (-4.0 / n)
        m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (n_heads - n), 2))
        m = torch.cat([m, m_hat])
    return 2 ** (n_heads // 2) * m


class RotaryMultiheadAttention(nn.MultiheadAttention):
    """Multihead attention with rotary positional embeddings.

    Extends PyTorch's MultiheadAttention with rotary positional embeddings (RoPE)
    for improved position-aware attention.

    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): Number of parallel attention heads.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        bias (bool, optional): If True, add bias to input/output projections. Defaults to True.
        add_bias_kv (bool, optional): Add bias to key/value projections. Defaults to False.
        add_zero_attn (bool, optional): Add zero attention. Defaults to False.
        kdim (int | None, optional): Key dimension. Defaults to None (=embed_dim).
        vdim (int | None, optional): Value dimension. Defaults to None (=embed_dim).
        batch_first (bool, optional): If True, input/output tensors are (batch, seq, feature).
            Defaults to False.
        device (torch.device | None, optional): Device for computation. Defaults to None.
        dtype (torch.dtype | None, optional): Data type for computation. Defaults to None.

    Attributes:
        rotary_emb (RotaryPositionalEmbeddings): Rotary position encoding module.

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: int | None = None,
        vdim: int | None = None,
        batch_first: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize rotary multihead attention module.

        Args:
            embed_dim (int): Total dimension of the model.
            num_heads (int): Number of parallel attention heads.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            bias (bool, optional): If True, add bias to input/output projections. Defaults to True.
            add_bias_kv (bool, optional): Add bias to key/value projections. Defaults to False.
            add_zero_attn (bool, optional): Add zero attention. Defaults to False.
            kdim (int | None, optional): Key dimension. Defaults to None (=embed_dim).
            vdim (int | None, optional): Value dimension. Defaults to None (=embed_dim).
            batch_first (bool, optional): If True, input/output tensors are (batch, seq, feature).
                Defaults to False.
            device (torch.device | None, optional): Device for computation. Defaults to None.
            dtype (torch.dtype | None, optional): Data type for computation. Defaults to None.

        """
        super().__init__(
            embed_dim,
            num_heads,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            kdim,
            vdim,
            batch_first,
            device,
            dtype,
        )
        self.rotary_emb = RotaryPositionalEmbeddings(embed_dim // num_heads, max_seq_len=20001)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = True,
        attn_mask: Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """Process input sequences through rotary multihead attention.

        Args:
            query (Tensor): Query embeddings (batch_size, seq_len, embed_dim).
            key (Tensor): Key embeddings (batch_size, seq_len, embed_dim).
            value (Tensor): Value embeddings (batch_size, seq_len, embed_dim).
            key_padding_mask (Tensor | None, optional): Mask for padded elements. Defaults to None.
            need_weights (bool, optional): If True, returns attention weights. Defaults to True.
            attn_mask (Tensor | None, optional): Mask to prevent attention to certain positions.
                Defaults to None.
            average_attn_weights (bool, optional): If True, returns averaged attention weights.
                Defaults to True.
            is_causal (bool, optional): If True, applies causal mask. Defaults to False.

        Returns:
            Tuple[Tensor, Optional[Tensor]]: Tuple of:
                - Output tensor of shape (batch_size, seq_len, embed_dim)
                - Attention weights if need_weights is True

        """
        # Reshape and apply rotary embeddings
        batch_size, seq_len, _ = query.shape
        query = query.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        query = self.rotary_emb(query)
        key = self.rotary_emb(key)

        # Reshape back
        query = query.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        key = key.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return super().forward(
            query,
            key,
            value,
            key_padding_mask,
            need_weights,
            attn_mask,
            average_attn_weights,
            is_causal,
        )


class ChromPositionalEncoding(nn.Module):
    """Chromatin-specific positional encoding for genomic sequences.

    Implements a specialized positional encoding that takes into account
    the unique properties of chromatin structure and genomic distances.

    Args:
        d_model (int): Dimension of the model.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        max_len (int, optional): Maximum sequence length. Defaults to 20001.

    Attributes:
        dropout (nn.Dropout): Dropout layer.
        d_model (int): Model dimension.
        pe (Tensor): Positional encoding buffer.

    """

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 20001) -> None:
        """Initialize chromatin-specific positional encoding.

        Args:
            d_model (int): Dimension of the model.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            max_len (int, optional): Maximum sequence length. Defaults to 20001.

        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model),
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply chromatin-specific positional encoding.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            positions (torch.Tensor): Genomic positions of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Encoded tensor of shape (batch_size, seq_len, d_model).

        """
        # Calculate chromatin positions efficiently
        chrom_positions = torch.zeros_like(positions, dtype=torch.long)
        chrom_positions[:, 1:] = (positions[:, 1:] - positions[:, :-1] > 10000).cumsum(dim=1)

        # Clamp and gather positional encodings
        chrom_positions = torch.clamp(chrom_positions, max=self.pe.size(0) - 1)
        pe_slice = self.pe[chrom_positions]

        # Add positional encoding to input
        return self.dropout(x + pe_slice)


class ChromosomePositionalEncoding(nn.Module):
    """Chromosome-aware positional encoding for genomic sequences.

    Implements positional encoding that incorporates chromosome information,
    allowing the model to distinguish between different chromosomes.

    Args:
        d_model (int): Dimension of the model.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        max_chromosomes (int, optional): Maximum number of chromosomes. Defaults to 50.

    Attributes:
        dropout (nn.Dropout): Dropout layer.
        d_model (int): Model dimension.
        max_chromosomes (int): Maximum number of chromosomes.
        pe (Tensor): Positional encoding buffer.

    """

    def __init__(self, d_model: int, dropout: float = 0.0, max_chromosomes: int = 50) -> None:
        """Initialize chromosome-aware positional encoding.

        Args:
            d_model (int): Dimension of the model.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            max_chromosomes (int, optional): Maximum number of chromosomes. Defaults to 50.

        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_chromosomes = max_chromosomes

        # Create positional encodings for each chromosome
        pe = torch.zeros(max_chromosomes, d_model)
        position = torch.arange(max_chromosomes, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, chromosomes: torch.Tensor) -> torch.Tensor:
        """Apply chromosome-aware positional encoding.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            chromosomes (torch.Tensor): Chromosome indices of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Encoded tensor of shape (batch_size, seq_len, d_model).

        """
        # Map chromosome labels to indices
        unique_chroms, chrom_indices = torch.unique(chromosomes, return_inverse=True)
        chrom_indices = chrom_indices.view(chromosomes.shape)

        # Clamp indices to max_chromosomes
        chrom_indices = torch.clamp(chrom_indices, max=self.max_chromosomes - 1)

        # Get positional encodings for each chromosome
        chrom_pe = self.pe[chrom_indices]

        # Add positional encoding to input
        return self.dropout(x + chrom_pe)


class AbsolutePositionalEncoding(nn.Module):
    """Genomic position-based positional encoding.

    Implements positional encoding based on absolute genomic coordinates
    rather than relative sequence positions.

    Args:
        d_embedding (int): Dimension of the embedding.
        max_position (int, optional): Maximum genomic position. Defaults to 300000000.
        dropout (float, optional): Dropout probability. Defaults to 0.0.

    Attributes:
        d_embedding (int): Embedding dimension.
        max_position (int): Maximum allowed genomic position.
        dropout (nn.Dropout): Dropout layer.

    """

    def __init__(
        self,
        d_embedding: int,
        max_position: int = 300000000,
        dropout: float = 0.0,
    ) -> None:
        """Initialize absolute positional encoding.

        Args:
            d_embedding (int): Dimension of the embedding.
            max_position (int, optional): Maximum genomic position. Defaults to 300000000.
            dropout (float, optional): Dropout probability. Defaults to 0.0.

        """
        super().__init__()

        self.d_embedding = d_embedding
        self.max_position = max_position
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply absolute positional encoding.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_embedding).
            positions (torch.Tensor): Genomic positions of shape (batch_size, seq_length).

        """
        # Normalize the positions to the range [0, 1]
        t = positions.float() / self.max_position  # Ensure positions are float for division

        bands = (self.d_embedding - 1) // 2
        w = 2 * math.pi * positions.float() / self.max_position  # Ensure positions are float

        # Generating frequency bands
        f = (
            torch.linspace(1e-4, bands - 1, bands, device=positions.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        # Calculate cosine and sine components
        cos_component = torch.cos(f * w.unsqueeze(-1))
        sin_component = torch.sin(f * w.unsqueeze(-1))

        # Concatenating the normalized positions with cosine and sine components
        z = torch.cat([t.unsqueeze(-1), t.unsqueeze(-1), cos_component, sin_component], dim=-1)

        return x + self.dropout(z)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding.

    Implements the standard transformer positional encoding using
    sine and cosine functions of different frequencies.

    Args:
        d_model (int): Dimension of the model.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        max_len (int, optional): Maximum sequence length. Defaults to 20001.

    Attributes:
        dropout (nn.Dropout): Dropout layer.
        pe (Tensor): Positional encoding buffer.

    """

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 20001) -> None:
        """Initialize standard positional encoding.

        Args:
            d_model (int): Dimension of the model.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            max_len (int, optional): Maximum sequence length. Defaults to 20001.

        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the forward pass of the dependencies module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the forward pass.

        """
        z = self.pe[:, : x.size(1)].expand(x.size(0), -1, -1).to(x.device)
        return x + self.dropout(z)


class SwiGLU(nn.Module):
    """SwiGLU module that applies the Swish-Gated Linear Unit (SwiGLU) activation function.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor after applying the SwiGLU activation function.

    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the dependencies module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the operation x1 * F.silu(x2).

        """
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * F.silu(x2)


class L2ScaleNorm(nn.Module):
    """Layer normalization using L2 norm with learned scaling parameter."""

    def __init__(self, d_hidden: int, eps: float = 1e-8) -> None:
        """Initialize L2ScaleNorm layer.

        Args:
            d_hidden (int): Hidden dimension size for the scaling parameter.
            eps (float, optional): Small value to prevent division by zero. Defaults to 1e-8.

        """
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_hidden))
        self.eps = eps
        self.d_hidden = d_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies L2 scaling normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor using L2 norm with learned scaling.

        """
        norm = torch.norm(x, dim=-1, keepdim=True)  # L2 norm
        scaled_norm = norm * self.scale / math.sqrt(self.d_hidden)  # Apply scaling
        return x / (scaled_norm + self.eps)


class MLPBlock(nn.Module):
    """Multi-Layer Perceptron (MLP) block module with optional normalization layers.

    Args:
        d_in (int): Input dimension.
        d_hidden (int): Hidden dimension.
        d_out (int): Output dimension.
        dropout (float): Dropout rate.
        n_blocks (int, optional): Number of blocks. Defaults to 1.
        expansion_factor (int, optional): Expansion factor. Defaults to 2.
        input_bias (bool, optional): Include bias in input layer. Defaults to True.
        bias (bool, optional): Include bias in hidden layers. Defaults to False.
        out_bias (bool, optional): Include bias in output layer. Defaults to False.
        activation (str, optional): Activation function. Defaults to "swiglu".
        pre_norm (bool, optional): Apply norm before input adapter. Defaults to True.
        post_norm (bool, optional): Apply norm before output adapter. Defaults to False.
        norm_type (str, optional): Type of normalization layer. Can be "layernorm", "rmsnorm",
            or "l2scalenorm". Defaults to "rmsnorm".

    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        dropout: float,
        n_blocks: int = 1,
        expansion_factor: int = 2,
        input_bias: bool = True,
        bias: bool = False,
        out_bias: bool = False,
        activation: str = "swiglu",
        pre_norm: bool = True,
        post_norm: bool = False,
        norm_type: str = "rmsnorm",
    ) -> None:
        """Multi-Layer Perceptron (MLP) block module with optional normalization layers.

        Args:
            d_in (int): Input dimension.
            d_hidden (int): Hidden dimension.
            d_out (int): Output dimension.
            dropout (float): Dropout rate.
            n_blocks (int, optional): Number of blocks. Defaults to 1.
            expansion_factor (int, optional): Expansion factor. Defaults to 2.
            input_bias (bool, optional): Include bias in input layer. Defaults to True.
            bias (bool, optional): Include bias in hidden layers. Defaults to False.
            out_bias (bool, optional): Include bias in output layer. Defaults to False.
            activation (str, optional): Activation function. Defaults to "swiglu".
            pre_norm (bool, optional): Apply norm before input adapter. Defaults to True.
            post_norm (bool, optional): Apply norm before output adapter. Defaults to False.
            norm_type (str, optional): Type of normalization layer. Can be "layernorm", "rmsnorm",
                or "l2scalenorm". Defaults to "rmsnorm".

        """
        super().__init__()
        self.activation = activation
        self.pre_norm = pre_norm
        self.post_norm = post_norm

        # Select normalization layer
        if norm_type == "rmsnorm":
            norm_layer = nn.RMSNorm
        elif norm_type == "layernorm":
            norm_layer = nn.LayerNorm
        elif norm_type == "l2scalenorm":
            norm_layer = L2ScaleNorm
        else:
            msg = f"Unsupported norm_type: {norm_type}"
            raise ValueError(msg)

        # Optional normalization before input adapter
        if self.pre_norm:
            self.input_norm = norm_layer(d_in)
        else:
            self.input_norm = nn.Identity()

        self.input_adapter = nn.Linear(d_in, d_hidden, bias=input_bias)

        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(
                nn.Sequential(
                    norm_layer(d_hidden),
                    nn.Linear(
                        d_hidden,
                        expansion_factor * d_hidden * (2 if activation == "swiglu" else 1),
                        bias=bias,
                    ),
                    _get_activation_fn(activation),
                    nn.Dropout(dropout),
                    nn.Linear(expansion_factor * d_hidden, d_hidden, bias=bias),
                ),
            )

        # Optional normalization before output adapter
        if self.post_norm:
            self.output_norm = norm_layer(d_hidden)
        else:
            self.output_norm = nn.Identity()

        self.output_adapter = nn.Linear(d_hidden, d_out, bias=out_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_in)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, d_out)

        """
        x = self.input_norm(x)
        x = self.input_adapter(x)
        for block in self.blocks:
            x = x + block(x)
        x = self.output_norm(x)
        return self.output_adapter(x)


class TransformerPPBlock(nn.Module):
    """Custom transformer encoder layer module.

    Args:
        d_model (int): Model dimension
        nhead (int): Number of attention heads
        dim_feedforward (int, optional): Feedforward dimension. Defaults to 128.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        activation (str, optional): Activation function. Defaults to "swiglu".
        expansion_factor (int, optional): Expansion factor. Defaults to 2.
        layer_norm_eps (float, optional): Layer norm epsilon. Defaults to 1e-8.
        batch_first (bool, optional): If True, batch is first dimension. Defaults to True.
        norm_first (bool, optional): If True, normalization before attention. Defaults to True.
        bias (bool, optional): If True, use bias in linear layers. Defaults to False.
        device (torch.device | None, optional): Computation device. Defaults to None.
        dtype (torch.dtype | None, optional): Data type. Defaults to None.
        fft (bool, optional): If True, use FFT mixing. Defaults to False.
        norm_type (str, optional): Type of normalization. Defaults to "rmsnorm".
        layer_dropout (float, optional): Dropout rate for layer. Defaults to 0.0.
        num_experts (int, optional): Number of experts. Defaults to 1.

    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        activation: str = "swiglu",
        expansion_factor: int = 2,
        layer_norm_eps: float = 1e-8,
        batch_first: bool = True,
        norm_first: bool = True,
        bias: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        fft: bool = False,
        norm_type: str = "rmsnorm",
        layer_dropout: float = 0.0,
        num_experts: int = 1,
    ) -> None:
        """Custom transformer encoder layer module.

        Args:
            d_model (int): Model dimension
            nhead (int): Number of attention heads
            dim_feedforward (int, optional): Feedforward dimension. Defaults to 128.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            activation (str, optional): Activation function. Defaults to "swiglu".
            expansion_factor (int, optional): Expansion factor. Defaults to 2.
            layer_norm_eps (float, optional): Layer norm epsilon. Defaults to 1e-8.
            batch_first (bool, optional): If True, batch is first dimension. Defaults to True.
            norm_first (bool, optional): If True, normalization before attention. Defaults to True.
            bias (bool, optional): If True, use bias in linear layers. Defaults to False.
            device (torch.device | None, optional): Computation device. Defaults to None.
            dtype (torch.dtype | None, optional): Data type. Defaults to None.
            fft (bool, optional): If True, use FFT mixing. Defaults to False.
            norm_type (str, optional): Type of normalization. Defaults to "rmsnorm".
            layer_dropout (float, optional): Dropout rate for layer. Defaults to 0.0.
            num_experts (int, optional): Number of experts. Defaults to 1.

        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.layer_dropout = layer_dropout
        self.num_experts = num_experts

        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            **factory_kwargs,
        )
        self.fft = fft
        if fft:
            self.fnet_mix = FNetMix()

        self.use_moe = num_experts > 1

        if self.use_moe:
            # Mixture of Experts
            self.moe = MixtureOfExperts(
                d_model,
                dim_feedforward,
                num_experts=num_experts,
                bias=bias,
                top_k=1,
                dropout=dropout,
                activation=activation,
                expansion_factor=expansion_factor,
            )
        else:
            self.linear1 = Linear(
                d_model,
                (
                    expansion_factor * dim_feedforward * 2
                    if activation == "swiglu"
                    else expansion_factor * dim_feedforward
                ),
                bias=bias,
                **factory_kwargs,
            )
            self.dropout_mlp = Dropout(dropout)
            self.linear2 = Linear(
                expansion_factor * dim_feedforward,
                d_model,
                bias=bias,
                **factory_kwargs,
            )

        # Select normalization layer
        if norm_type == "rmsnorm":
            norm_layer = nn.RMSNorm
        elif norm_type == "layernorm":
            norm_layer = nn.LayerNorm
        elif norm_type == "l2scalenorm":
            norm_layer = L2ScaleNorm
        else:
            msg = f"Unsupported norm_type: {norm_type}"
            raise ValueError(msg)

        self.norm_first = norm_first
        self.norm1_cls = norm_layer(d_model)
        self.norm1_seq = norm_layer(d_model)
        self.norm2_cls = norm_layer(d_model)
        self.norm2_seq = norm_layer(d_model)
        self.dropout_sa = Dropout(dropout)
        self.dropout_ff = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Applies forward pass of the dependencies module."""
        if self.training and torch.rand(1).item() < self.layer_dropout:
            return src * (1.0 / (1.0 - self.layer_dropout))

        x = src
        if self.norm_first:
            x_sa = self._sa_block(self._split_norm1(x), src_mask, src_key_padding_mask)
            x = x + x_sa
            x_ff = self._ff_block(self._split_norm2(x))
            x = x + x_ff
        else:
            x_sa = self._sa_block(x, src_mask, src_key_padding_mask)
            x = self._split_norm1(x + x_sa)
            x_ff = self._ff_block(x)
            x = self._split_norm2(x + x_ff)
        return x

    def _split_norm1(self, x: torch.Tensor) -> torch.Tensor:
        """Splits the input tensor and applies layer normalization."""
        return torch.cat([self.norm1_cls(x[:, :1]), self.norm1_seq(x[:, 1:])], dim=1)

    def _split_norm2(self, x: torch.Tensor) -> torch.Tensor:
        """Splits the input tensor and applies layer normalization."""
        return torch.cat([self.norm2_cls(x[:, :1]), self.norm2_seq(x[:, 1:])], dim=1)

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Applies the self-attention block."""
        if self.fft:
            x = self.fnet_mix(x)
        else:
            x = self.self_attn(
                x,
                x,
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )[0]
        return self.dropout_sa(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the feedforward block."""
        if self.use_moe:
            x = self.moe(x)
        else:
            x = self.dropout_ff(self.linear2(self.dropout_mlp(self.activation(self.linear1(x)))))
        return x


def _get_activation_fn(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Returns the activation function based on the given string."""
    if activation == "relu":
        return nn.ReLU()
    if activation == "gelu":
        return nn.GELU()
    if activation == "swiglu":
        return SwiGLU()
    msg = f"activation should be relu/gelu/swiglu, not {activation}"
    raise RuntimeError(msg)


class MixtureOfExperts(nn.Module):
    """Mixture of Experts (MoE) layer with load balancing and capacity control.

    This implementation includes:
    - Top-k expert routing with noise injection for exploration
    - Load balancing via capacity constraints and auxiliary loss
    - Temperature-scaled gating
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        num_experts: int = 4,
        top_k: int = 1,
        capacity_factor: float = 1.25,
        noise_scale: float = 0.1,
        aux_loss_weight: float = 0.01,
        temperature: float = 0.5,
        dropout: float = 0.1,
        activation: str = "gelu",
        expansion_factor: int = 2,
        bias: bool = True,
    ) -> None:
        """Initialize the MoE layer.

        Args:
            d_model: Input/output dimension
            d_hidden: Hidden dimension of expert MLPs
            num_experts: Number of expert networks
            top_k: Number of experts to route each input to
            capacity_factor: Multiplier for expert capacity
            noise_scale: Scale of noise added to gating logits
            aux_loss_weight: Weight of load balancing auxiliary loss
            temperature: Temperature for gating softmax
            dropout: Dropout probability
            activation: Activation function to use
            expansion_factor: Expansion factor for expert MLPs
            bias: If True, use bias in linear layers

        """
        super().__init__()

        # Core parameters
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        # Gating parameters
        self.noise_scale = noise_scale
        self.temperature = temperature

        # Loss parameters
        self.aux_loss_weight = aux_loss_weight

        # Expert networks - each is a feedforward block
        self.experts = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "linear1": nn.Linear(
                            d_model,
                            d_hidden * expansion_factor * 2
                            if activation == "swiglu"
                            else d_hidden * expansion_factor,
                            bias=bias,
                        ),
                        "linear2": nn.Linear(d_hidden * expansion_factor, d_model, bias=bias),
                        "activation": _get_activation_fn(activation),
                        "dropout_mlp": nn.Dropout(dropout),
                        "dropout_ff": nn.Dropout(dropout),
                    }
                )
                for _ in range(num_experts)
            ]
        )

        # Gating network
        self.gate = nn.Sequential(nn.Linear(d_model, num_experts), nn.RMSNorm(num_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass routing inputs through experts.

        Args:
            x: Input tensor of shape [batch, seq_len, d_model]

        Returns:
            Combined expert outputs of shape [batch, seq_len, d_model]

        """
        # Reshape input
        original_shape = x.shape
        x = x.view(-1, original_shape[-1])  # [batch*seq, d_model]

        # Get expert assignments
        gates, top_k_indices = self._compute_gating(x)

        # Apply capacity constraints
        expert_capacity = self._compute_capacity(x.size(0))
        dispatch_mask = self._create_dispatch_mask(top_k_indices, expert_capacity)

        # Normalize gates after masking
        masked_gates = gates * dispatch_mask
        normalized_gates = masked_gates / (masked_gates.sum(dim=-1, keepdim=True) + 1e-6)

        # Process inputs through experts
        combined_output = self._process_experts(x, normalized_gates)
        return combined_output.view(*original_shape)

    def _compute_gating(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute gating logits and select top-k experts."""
        # Get raw gates and add noise during training
        logits = self.gate(x)
        if self.training:
            logits += torch.randn_like(logits) * self.noise_scale

        # Temperature-scaled softmax
        gates = F.softmax(logits / self.temperature, dim=-1)

        # Select top-k experts
        top_k_gates, top_k_indices = torch.topk(gates, self.top_k, dim=-1)
        return top_k_gates, top_k_indices

    def _compute_capacity(self, batch_size: int) -> int:
        """Compute expert capacity based on batch size."""
        return min(
            math.ceil((batch_size * self.top_k) / self.num_experts * self.capacity_factor),
            batch_size,
        )

    def _process_experts(self, inputs: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        """Process inputs through experts and combine outputs."""
        expert_inputs = inputs.repeat_interleave(self.top_k, dim=0)
        expert_outputs = []
        expert_outputs = torch.stack(
            [
                expert["dropout_ff"](
                    expert["linear2"](
                        expert["dropout_mlp"](
                            expert["activation"](expert["linear1"](expert_inputs))
                        )
                    )
                )
                for expert in self.experts
            ],
            dim=1,
        )
        return (expert_outputs * gates.view(-1, 1, 1)).sum(dim=1)

    def _get_raw_gates(self, x: torch.Tensor) -> torch.Tensor:
        """Get raw gating probabilities."""
        return F.softmax(self.gate(x) / self.temperature, dim=-1)

    def _create_dispatch_mask(
        self, expert_indices: torch.Tensor, expert_capacity: int
    ) -> torch.Tensor:
        """Create dispatch mask respecting capacity constraints."""
        mask = torch.zeros_like(expert_indices, dtype=torch.float32)
        for expert_idx in range(self.num_experts):
            expert_mask = expert_indices == expert_idx
            counts = expert_mask.sum(dim=0)
            capacity_mask = torch.arange(expert_capacity, device=expert_indices.device).expand(
                len(counts), -1
            ) < counts.unsqueeze(-1)
            mask += expert_mask.float() * capacity_mask.any(dim=-1).float()
        return mask


# Gradient Reversal Layer
class GradientReversal(torch.autograd.Function):
    """Gradient reversal layer for domain adaptation."""

    @staticmethod
    def forward(ctx: torch.autograd.Function, x: torch.Tensor, alpha: float) -> torch.Tensor:
        """Forward pass of the gradient reversal layer.

        Args:
            ctx (torch.autograd.Function): Autograd context
            x (torch.Tensor): Input tensor
            alpha (float): Scaling factor for gradient reversal

        Returns:
            torch.Tensor: Input tensor without modification

        """
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(
        ctx: torch.autograd.Function,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        """Backward pass of the gradient reversal layer.

        Args:
            ctx (torch.autograd.Function): Autograd context containing alpha value
            grad_output (torch.Tensor): Gradient from subsequent layer

        Returns:
            tuple[torch.Tensor, None]: Tuple containing negated gradient and None for alpha

        """
        return grad_output.neg() * ctx.alpha, None


def gradient_reversal(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Applies gradient reversal to the input tensor.

    Args:
        x (torch.Tensor): Input tensor to apply gradient reversal to
        alpha (float, optional): Scaling factor for gradient reversal. Defaults to 1.0.

    Returns:
        torch.Tensor: Tensor with reversed gradients during backpropagation

    """
    return GradientReversal.apply(x, alpha)


class FNetMix(nn.Module):
    """Fast-Fourier Transform."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies FFT transform across dimensions 1 and 2 (batch, seq, dim).

        Args:
            x: Input tensor of shape (batch, seq, dim)

        Returns:
            torch.Tensor: FFT transformed tensor

        """
        fft_hidden = torch.fft.fft(x, dim=2)
        fft_seq = torch.fft.fft(fft_hidden, dim=1)
        return torch.real(fft_seq)
