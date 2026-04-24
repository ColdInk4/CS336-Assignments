# FLOPs by matmul per sequence: 6 * d_model * d_ff * sequence_length

import torch
from torch import nn, Tensor
from jaxtyping import Float
from cs336_basics.Transformer import Linear


def SiLU(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        # FLOPs by matmul per sequence: 6 * d_model * d_ff * sequence_length

        w1_x = self.w1(x)
        w3_x = self.w3(x)
        return self.w2(SiLU(w1_x) * w3_x)
