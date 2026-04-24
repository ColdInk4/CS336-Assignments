# FLOPs by matmul per sequence:
# num_layers
# * (8 * d_model * d_model * sequence_length + 4 * sequence_length^2 * d_model + 6 * d_model * d_ff * sequence_length)
# + 2 * d_model * vocab_size * sequence_length

# trainable parameters:
# 2 * vocab_size * d_model
# + num_layers * (4 * d_model * d_model + 2 * d_model + 3 * d_model * d_ff)
# + d_model
import torch
from torch import nn, Tensor
from jaxtyping import Float, Int
from cs336_basics.Transformer import (
    Embedding,
    Transformer_Block,
    RMSNorm,
    Linear,
)


class Transformer_LM(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        # trainable parameters: vocab_size * d_model
        self.token_embeddings = Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )
        # trainable parameters: num_layers * (4 * d_model * d_model + 2 * d_model + 3 * d_model * d_ff)
        self.layers = nn.ModuleList(
            Transformer_Block(
                d_model,
                num_heads,
                d_ff,
                max_seq_len=context_length,
                theta=theta,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        )
        # trainable parameters: d_model
        self.ln_final = RMSNorm(
            d_model,
            device=device,
            dtype=dtype,
        )
        # trainable parameters: d_model * vocab_size
        self.lm_head = Linear(
            d_model,
            vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self, x: Int[Tensor, "batch_size sequence_length"]
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        embedding_values: Float[Tensor, "batch_size sequence_length d_model"] = (
            self.token_embeddings(x)
        )
        transformer_values: Float[Tensor, "batch_size sequence_length d_model"] = (
            embedding_values
        )

        # FLOPs by matmul per sequence: num_layers * (8 * d_model * d_model * sequence_length + 4 * sequence_length^2 * d_model + 6 * d_model * d_ff * sequence_length)
        for transformer_block in self.layers:
            transformer_values = transformer_block(transformer_values)

        norm_values: Float[Tensor, "batch_size sequence_length d_model"] = (
            self.ln_final(transformer_values)
        )

        # FLOPs by matmul per sequence: 2 * d_model * vocab_size * sequence_length
        lm_head_values: Float[Tensor, "batch_size sequence_length vocab_size"] = (
            self.lm_head(norm_values)
        )
        return lm_head_values
