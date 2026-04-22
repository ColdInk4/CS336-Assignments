import torch
from torch import nn, Tensor
from jaxtyping import Float


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight: Float[Tensor, "d_model"] = nn.Parameter(
            torch.ones(self.d_model, device=device, dtype=dtype)
        )

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms: Float[Tensor, "... 1"] = ((x**2).mean(-1, keepdim=True) + self.eps).sqrt()

        output = x / rms * self.weight

        return output.to(in_dtype)
