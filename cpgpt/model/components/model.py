import math

import torch
import torch.nn.functional as F
from torch import nn

try:
    from mamba_ssm import Mamba2

    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False

from torchtune.modules import RotaryPositionalEmbeddings

from cpgpt.model.utils import m_to_beta

from .modules import (
    AbsolutePositionalEncoding,
    ChromosomePositionalEncoding,
    L2ScaleNorm,
    MLPBlock,
    TransformerPPBlock,
    create_hic_attention_mask,
)


class CpGPT(nn.Module):
    """A deep learning model for CpG methylation prediction.

    This model implements a flexible architecture for DNA methylation prediction,
    supporting both transformer and mamba-based architectures. It can process DNA
    sequences and their methylation states, with optional condition prediction.

    Args:
        d_embedding (int): Dimension of the embedding space.
        d_hidden (int): Dimension of hidden layers.
        d_dna_embedding (int): Dimension of DNA embeddings.
        n_attention_heads (int): Number of attention heads.
        n_layers (int): Number of transformer/mamba layers.
        n_mlp_blocks (int): Number of MLP blocks.
        dropout (float): Dropout rate.
        architecture (str): Model architecture type ("mamba" or "transformer").
        activation (str): Activation function name.
        positional_encoding (str): Type of positional encoding.
        sample_embedding_method (str): Method for sample embedding ('cls', 'mean' or 'max').
        use_power_norm (bool): Whether to use power normalization.
        fft (bool): Whether to use FFT instead of attention.
        use_condition_decoder (bool): Whether to use condition decoder.
        condition_size (int): Size of query for condition decoder.
        mlp_block_bias (bool): Whether to use bias in MLP blocks
        mlp_block_norm_type (str): Type of normalization for MLP blocks
        mlp_block_pre_norm (bool): Whether to apply normalization before the input adapter
            in MLP blocks
        mlp_block_post_norm (bool): Whether to apply normalization before the output adapter
            in MLP blocks
        transformer_block_bias (bool): Whether to use bias in transformer blocks
        transformer_block_norm_type (str): Type of normalization for transformer blocks
        transformer_block_norm_first (bool): Whether to apply normalization before the
            self-attention in transformer blocks
        transformer_block_dropout (float): Dropout rate for transformer blocks

    Attributes:
        cls_token (nn.Parameter): Learnable classification token.
        position_encoder: Positional encoding module.
        dna_encoder (MLPBlock): DNA sequence encoder.
        meth_encoder (MLPBlock): Methylation value encoder.
        transformer_encoder (Optional[nn.TransformerEncoder]): Transformer encoder if used.
        mamba_fwd_encoder (Optional[nn.ModuleList]): Forward Mamba layers if used.
        mamba_rev_encoder (Optional[nn.ModuleList]): Reverse Mamba layers if used.
        meth_decoder (MLPBlock): Methylation prediction decoder.
        meth_unc_decoder (MLPBlock): Methylation uncertainty decoder.
        condition_decoder (Optional[MLPBlock]): Condition prediction decoder if used.

    """

    def __init__(
        self,
        d_embedding: int,
        d_hidden: int,
        d_dna_embedding: int,
        n_attention_heads: int,
        n_layers: int,
        n_mlp_blocks: int,
        dropout: float,
        architecture: str,
        activation: str,
        positional_encoding: str,
        sample_embedding_method: str,
        use_power_norm: bool,
        fft: bool,
        use_condition_decoder: bool,
        condition_size: int,
        mlp_block_bias: bool,
        mlp_block_norm_type: str,
        mlp_block_pre_norm: bool,
        mlp_block_post_norm: bool,
        transformer_block_bias: bool,
        transformer_block_norm_type: str,
        transformer_block_norm_first: bool,
        transformer_block_dropout: float,
        **kwargs: object,
    ) -> None:
        """Initialize the CpGPT model.

        Args:
            d_embedding: Dimension of the embedding space
            d_hidden: Dimension of hidden layers
            d_dna_embedding: Dimension of DNA embeddings
            n_attention_heads: Number of attention heads
            n_layers: Number of transformer/mamba layers
            n_mlp_blocks: Number of MLP blocks
            dropout: Dropout rate
            architecture: Model architecture type ("mamba" or "transformer")
            activation: Activation function name
            positional_encoding: Type of positional encoding
            sample_embedding_method: Method for sample embedding ('cls', 'mean' or 'max')
            use_power_norm: Whether to use power normalization
            fft: Whether to use FFT instead of attention
            use_condition_decoder: Whether to use condition decoder
            condition_size: Size of query for condition decoder
            mlp_block_bias: Whether to use bias in MLP blocks
            mlp_block_norm_type: Type of normalization for MLP blocks
            mlp_block_pre_norm: Whether to apply normalization before the input adapter
                in MLP blocks
            mlp_block_post_norm: Whether to apply normalization before the output adapter
                in MLP blocks
            transformer_block_bias: Whether to use bias in transformer blocks
            transformer_block_norm_type: Type of normalization for transformer blocks
            transformer_block_norm_first: Whether to apply normalization before the
                self-attention in transformer blocks
            transformer_block_dropout: Dropout rate for transformer blocks

        """
        super().__init__()

        # Backwards-compatibility: older configs may include legacy keys that are no longer used.
        # We intentionally ignore them.
        if "use_noise_decoder" in kwargs:
            kwargs.pop("use_noise_decoder")
        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected kwargs for CpGPT: {unexpected}")

        # Store initialization parameters
        self.d_embedding = d_embedding
        self.d_hidden = d_hidden
        self.d_dna_embedding = d_dna_embedding
        self.n_attention_heads = n_attention_heads
        self.n_layers = n_layers
        self.n_mlp_blocks = n_mlp_blocks
        self.dropout = dropout
        self.architecture = architecture
        self.activation = activation
        self.positional_encoding = positional_encoding
        self.sample_embedding_method = sample_embedding_method
        self.use_power_norm = use_power_norm
        self.fft = fft
        self.use_condition_decoder = use_condition_decoder
        self.condition_size = condition_size
        self.mlp_block_bias = mlp_block_bias
        self.mlp_block_norm_type = mlp_block_norm_type
        self.mlp_block_pre_norm = mlp_block_pre_norm
        self.mlp_block_post_norm = mlp_block_post_norm
        self.transformer_block_bias = transformer_block_bias
        self.transformer_block_norm_type = transformer_block_norm_type
        self.transformer_block_norm_first = transformer_block_norm_first
        self.transformer_block_dropout = transformer_block_dropout

        # Positional encoder
        if self.positional_encoding == "rotary":
            self.position_encoder = RotaryPositionalEmbeddings(
                self.d_embedding // self.n_attention_heads,
                max_seq_len=20001,
            )
            self.absolute_position_encoder = AbsolutePositionalEncoding(
                self.d_embedding,
                dropout=self.dropout,
            )
        elif self.positional_encoding == "positional":
            self.position_encoder = ChromosomePositionalEncoding(
                self.d_embedding,
                dropout=self.dropout,
            )
            self.absolute_position_encoder = AbsolutePositionalEncoding(
                self.d_embedding,
                dropout=self.dropout,
            )

        # DNA sequence encoder
        self.dna_encoder = MLPBlock(
            self.d_dna_embedding,
            self.d_hidden,
            self.d_embedding,
            self.dropout,
            n_blocks=self.n_mlp_blocks,
            activation=self.activation,
            bias=self.mlp_block_bias,
            norm_type=self.mlp_block_norm_type,
            pre_norm=self.mlp_block_pre_norm,
            post_norm=self.mlp_block_post_norm,
        )

        # Methylation encoder
        self.meth_encoder = MLPBlock(
            1,
            self.d_hidden,
            self.d_embedding,
            self.dropout,
            n_blocks=1,
            activation=self.activation,
            bias=self.mlp_block_bias,
            norm_type=self.mlp_block_norm_type,
            pre_norm=self.mlp_block_pre_norm,
            post_norm=self.mlp_block_post_norm,
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_embedding))

        # Main processing blocks
        if "transformer" in self.architecture:
            encoder_layer = TransformerPPBlock(
                self.d_embedding,
                self.n_attention_heads,
                self.d_hidden,
                batch_first=True,
                bias=self.transformer_block_bias,
                activation=self.activation,
                norm_first=self.transformer_block_norm_first,
                fft=self.fft,
                norm_type=self.transformer_block_norm_type,
                dropout=self.transformer_block_dropout,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=self.n_layers,
                enable_nested_tensor=False,
            )
        elif "mamba" in self.architecture:
            self.mamba_fwd_encoder = nn.ModuleList(
                [
                    Mamba2(
                        d_model=self.d_embedding,
                        headdim=self.d_embedding // 4,
                        d_state=64,
                        d_conv=4,
                        expand=2,
                    )
                    for _ in range(self.n_layers)
                ],
            )
            self.mamba_rev_encoder = nn.ModuleList(
                [
                    Mamba2(
                        d_model=self.d_embedding,
                        headdim=self.d_embedding // 4,
                        d_state=64,
                        d_conv=4,
                        expand=2,
                    )
                    for _ in range(self.n_layers)
                ],
            )

        # Methylation decoder
        self.meth_decoder = MLPBlock(
            self.d_embedding,
            self.d_hidden,
            self.d_embedding,
            self.dropout,
            n_blocks=self.n_mlp_blocks,
            out_bias=False,
            activation=self.activation,
            bias=self.mlp_block_bias,
            norm_type=self.mlp_block_norm_type,
            pre_norm=self.mlp_block_pre_norm,
            post_norm=self.mlp_block_post_norm,
        )

        # Methylation uncertainty decoder
        self.meth_unc_decoder = MLPBlock(
            self.d_embedding,
            self.d_hidden,
            self.d_embedding,
            self.dropout,
            n_blocks=self.n_mlp_blocks,
            out_bias=False,
            activation=self.activation,
            bias=self.mlp_block_bias,
            norm_type=self.mlp_block_norm_type,
            pre_norm=self.mlp_block_pre_norm,
            post_norm=self.mlp_block_post_norm,
        )

        # Conditions-specific decoder
        if self.use_condition_decoder:
            self.condition_tokens = nn.ParameterList(
                [
                    nn.Parameter(torch.randn(1, self.d_embedding))
                    for _ in range(self.condition_size)
                ],
            )
            self.condition_decoder = MLPBlock(
                self.d_embedding,
                self.d_hidden,
                self.d_embedding,
                self.dropout,
                n_blocks=self.n_mlp_blocks,
                out_bias=True,
                activation=self.activation,
                bias=self.mlp_block_bias,
                norm_type=self.mlp_block_norm_type,
                pre_norm=self.mlp_block_pre_norm,
                post_norm=self.mlp_block_post_norm,
            )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights using appropriate initialization schemes.

        Applies:
            - Ones initialization for normalization layer scales
            - Zeros initialization for normalization layer biases
            - Normal initialization for embedding tokens
        """
        for m in self.modules():
            if isinstance(m, nn.RMSNorm | nn.LayerNorm | L2ScaleNorm):
                if hasattr(m, "scale"):
                    nn.init.ones_(m.scale)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.1)
        if self.use_condition_decoder:
            for token in self.condition_tokens:
                nn.init.normal_(token, mean=0.0, std=0.1)

    def encode_sequence(self, dna_embeddings: torch.Tensor) -> torch.Tensor:
        """Encode DNA sequence embeddings into a higher-level representation.

        Args:
            dna_embeddings (torch.Tensor): DNA embeddings tensor of shape
                (batch_size, seq_len, d_dna_embedding).

        Returns:
            torch.Tensor: Encoded sequence embeddings of shape
                (batch_size, seq_len, d_embedding).

        """
        return self.dna_encoder(dna_embeddings)

    def encode_sample(
        self,
        meth: torch.Tensor,
        sequence_embeddings: torch.Tensor,
        chroms: torch.Tensor,
        positions: torch.Tensor,
        mask_na: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a methylation sample into a fixed-size representation.

        This method combines sequence embeddings with methylation values and positional
        information to create a comprehensive sample representation.

        Args:
            meth (torch.Tensor): Methylation beta values of shape (batch_size, seq_len).
            sequence_embeddings (torch.Tensor): Sequence embeddings of shape
                (batch_size, seq_len, d_embedding).
            chroms (torch.Tensor): Chromosome indices of shape (batch_size, seq_len).
            positions (torch.Tensor): Chromosomal positions of shape (batch_size, seq_len).
            mask_na (torch.Tensor): Boolean mask for missing values of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Sample embedding of shape (batch_size, d_embedding).

        """
        # Apply positional encoding
        if self.positional_encoding == "positional":
            locus_embeddings = self.absolute_position_encoder(sequence_embeddings, positions)
            locus_embeddings = self.position_encoder(locus_embeddings, chroms)
        elif self.positional_encoding == "rotary":
            locus_embeddings = self.absolute_position_encoder(sequence_embeddings, positions)
            batch_size, seq_len, _ = locus_embeddings.shape
            locus_embeddings = locus_embeddings.view(
                batch_size,
                seq_len,
                self.n_attention_heads,
                -1,
            ).transpose(1, 2)
            locus_embeddings = self.position_encoder(locus_embeddings)
            locus_embeddings = (
                locus_embeddings.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            )
        else:
            locus_embeddings = sequence_embeddings

        # Methylation encoding
        meth_embedding = self.meth_encoder(meth.unsqueeze(-1))

        # Combine locus and methylation embeddings
        cpg_embeddings = locus_embeddings + meth_embedding
        cpg_embeddings = cpg_embeddings / math.sqrt(2)  # scale variance

        # Add CLS token
        cls_token = self.cls_token.expand(cpg_embeddings.size(0), -1, -1)
        cpg_embeddings = torch.cat([cls_token, cpg_embeddings], dim=1)
        mask_na = torch.cat(
            [torch.zeros(mask_na.size(0), 1, dtype=torch.bool, device=mask_na.device), mask_na],
            dim=1,
        )

        # Set masked positions to 0
        cpg_embeddings[mask_na] = 0

        # Processing blocks
        if "mamba" in self.architecture:
            for layer_fwd, layer_rev in zip(
                self.mamba_fwd_encoder,
                self.mamba_rev_encoder,
                strict=False,
            ):
                cpg_embeddings_fwd = layer_fwd(cpg_embeddings)
                cpg_embeddings_rev = layer_rev(cpg_embeddings.flip((1,))).flip((1,))
                cpg_embeddings = cpg_embeddings_fwd + cpg_embeddings_rev

        if "transformer" in self.architecture:
            attn_mask = (
                create_hic_attention_mask(chroms, positions, self.n_attention_heads)
                if self.positional_encoding == "hic"
                else None
            )
            cpg_embeddings = self.transformer_encoder(
                cpg_embeddings,
                src_key_padding_mask=mask_na.bool(),
                mask=attn_mask,
            )

        # Get sample embedding
        if (
            self.training
            and self.sample_embedding_method == "cls"
            and not self.use_condition_decoder
        ):
            if torch.rand(1) < 0.5:
                sample_embedding = cpg_embeddings[:, 0, :]
            else:
                sample_embedding = cpg_embeddings[:, 1:, :].mean(dim=1)
        elif self.sample_embedding_method == "cls":
            sample_embedding = cpg_embeddings[:, 0, :]
        elif self.sample_embedding_method == "mean":
            sample_embedding = cpg_embeddings[:, 1:, :].mean(dim=1)
        elif self.sample_embedding_method == "max":
            sample_embedding = cpg_embeddings[:, 1:, :].max(dim=1)[0]

        # Power normalization
        if self.use_power_norm:
            sample_embedding = torch.sign(sample_embedding) * torch.abs(sample_embedding).pow(0.5)
            sample_embedding = F.normalize(sample_embedding, p=2, dim=-1)

        return sample_embedding

    def query_methylation(
        self,
        sample_embedding: torch.Tensor,
        sequence_embeddings: torch.Tensor,
        m_or_beta: str = "m",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict methylation values and their uncertainties for given sequences.

        Args:
            sample_embedding (torch.Tensor): Sample embedding of shape (batch_size, d_embedding).
            sequence_embeddings (torch.Tensor): Sequence embeddings of shape
                (batch_size, seq_len, d_embedding).
            m_or_beta (str, optional): Output format, either "m" for M-values or
                "beta" for beta-values. Defaults to "m".

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Predicted methylation values of shape (batch_size, seq_len)
                - Predicted uncertainties of shape (batch_size, seq_len)

        """
        # Combine the attention output with the original sequence embeddings
        pred_meth = torch.bmm(
            self.meth_decoder(sequence_embeddings),
            sample_embedding.unsqueeze(2),
        ).squeeze(2)

        pred_meth_unc = F.softplus(
            torch.bmm(
                self.meth_unc_decoder(sequence_embeddings),
                sample_embedding.unsqueeze(2),
            ).squeeze(2),
        )

        if m_or_beta == "beta":
            pred_meth = m_to_beta(pred_meth)
            pred_meth_unc = m_to_beta(pred_meth_unc)

        return pred_meth, pred_meth_unc

    def query_condition(self, sample_embedding: torch.Tensor) -> torch.Tensor:
        """Predict condition values from a sample embedding.

        Args:
            sample_embedding (torch.Tensor): Sample embedding of shape (batch_size, d_embedding).

        Returns:
            torch.Tensor: Predicted condition values of shape (batch_size, condition_size).

        """
        # Stack all condition tokens and process them in parallel
        stacked_tokens = torch.stack(
            list(self.condition_tokens),
            dim=1,
        )  # Shape: (1, num_conditions, d_embedding)
        processed_tokens = self.condition_decoder(
            stacked_tokens,
        )  # Shape: (1, num_conditions, d_embedding)

        # Expand across the batch dimension
        condition_tokens = processed_tokens.expand(
            sample_embedding.size(0),
            -1,
            -1,
        )  # Shape: (batch_size, num_conditions, embed_dim)

        # Perform batch matrix multiplication
        return torch.bmm(condition_tokens, sample_embedding.unsqueeze(2)).squeeze(2)

    def forward(
        self,
        meth: torch.Tensor,
        dna_embeddings: torch.Tensor,
        chroms: torch.Tensor,
        positions: torch.Tensor,
        mask_na: torch.Tensor | None = None,
    ) -> dict:
        """Process a batch of methylation data through the model.

        Args:
            meth (torch.Tensor): Methylation beta values of shape (batch_size, seq_len).
            dna_embeddings (torch.Tensor): DNA embeddings of shape
                (batch_size, seq_len, d_dna_embedding).
            chroms (torch.Tensor): Chromosome indices of shape (batch_size, seq_len).
            positions (torch.Tensor): Chromosomal positions of shape (batch_size, seq_len).
            mask_na (Optional[torch.Tensor], optional): Boolean mask for missing values,
                shape (batch_size, seq_len). Defaults to None.

        Returns:
            dict: Dictionary containing:
                - sample_embedding: Sample representation (batch_size, d_embedding)
                - pred_meth: Predicted methylation values (batch_size, seq_len)
                - pred_meth_unc: Predicted uncertainties (batch_size, seq_len)
                - pred_conditions: Optional predicted conditions if enabled

        """
        output = {}

        sequence_embeddings = self.encode_sequence(dna_embeddings)

        sample_embedding = self.encode_sample(
            sequence_embeddings,
            meth,
            chroms,
            positions,
            mask_na,
        )
        output["sample_embedding"] = sample_embedding

        pred_meth, pred_meth_unc = self.query_methylation(sample_embedding, sequence_embeddings)
        output["pred_meth"] = pred_meth
        output["pred_meth_unc"] = pred_meth_unc

        if self.use_condition_decoder:
            pred_conditions = self.query_condition(sample_embedding)
            output["pred_conditions"] = pred_conditions

        return output
