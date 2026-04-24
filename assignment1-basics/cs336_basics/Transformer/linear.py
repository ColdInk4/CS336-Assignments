import torch
from torch import nn, Tensor
from math import sqrt
from einops import einsum
from jaxtyping import Float


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight: Float[Tensor, " d_out d_in"] = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        init_mean = 0
        init_std = sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(
            self.weight,
            mean=init_mean,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )

    def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        return einsum(x, self.weight, "... d_in, d_out d_in ->... d_out")
