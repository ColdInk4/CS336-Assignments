# FLOPs by matmul: 0
# trainable parameters: vocab_size * d_model

import torch
from torch import nn, Tensor
from jaxtyping import Float, Int64


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight: Float[Tensor, " vocab_size d_model"] = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        init_mean = 0
        init_std = 1
        nn.init.trunc_normal_(
            self.weight,
            mean=init_mean,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )

    def forward(self, token_ids: Int64[Tensor, "..."]) -> Float[Tensor, "... d_model"]:

        return self.weight[token_ids]
