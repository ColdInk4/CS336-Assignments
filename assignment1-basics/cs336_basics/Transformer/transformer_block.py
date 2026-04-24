import torch
from torch import nn, Tensor
from jaxtyping import Float
from cs336_basics.Transformer import Multihead_Self_Attention, RMSNorm, SwiGLU


class Transformer_Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.attn = Multihead_Self_Attention(
            d_model,
            num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        output1 = x + self.attn(self.ln1(x))
        result = output1 + self.ffn(self.ln2(output1))
        return result
