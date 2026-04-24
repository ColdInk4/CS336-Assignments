import torch
from torch import nn, Tensor
from jaxtyping import Float, Int
from cs336_basics.Transformer import (
    Embedding,
    Transformer_Block,
    RMSNorm,
    Linear,
    softmax,
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
        self.token_embeddings = Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )
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
        self.ln_final = RMSNorm(
            d_model,
            device=device,
            dtype=dtype,
        )
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
        for transformer_block in self.layers:
            transformer_values = transformer_block(transformer_values)

        norm_values: Float[Tensor, "batch_size sequence_length d_model"] = (
            self.ln_final(transformer_values)
        )

        lm_head_values: Float[Tensor, "batch_size sequence_length vocab_size"] = (
            self.lm_head(norm_values)
        )
        return lm_head_values
