# FLOPs by matmul per sequence across all heads: 8 * d_model * d_model * sequence_length + 4 * sequence_length^2 * d_model

import torch
from torch import nn, Tensor
from einops import rearrange
from jaxtyping import Float
from cs336_basics.Transformer import Linear, scaled_dot_product_attention, RoPE


class Multihead_Self_Attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        # 我们本次的实现里，hd_k = hd_v = d_model/num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = None
        if max_seq_len is not None and theta is not None:
            self.rope = RoPE(
                theta, int(d_model / num_heads), max_seq_len, device=device, dtype=dtype
            )

    # self_attention 里 queries = keys = sequence_length
    def multihead(
        self,
        Q: Float[Tensor, "... sequence_length d_k"],
        K: Float[Tensor, "... sequence_length d_k"],
        V: Float[Tensor, "... sequence_length d_v"],
    ) -> Float[Tensor, "... sequence_length d_v"]:
        Q_slices = rearrange(
            Q,
            "... sequence_length (num_heads hd_k) -> ... num_heads sequence_length hd_k",
            num_heads=self.num_heads,
        )
        K_slices = rearrange(
            K,
            "... sequence_length (num_heads hd_k) -> ... num_heads sequence_length hd_k",
            num_heads=self.num_heads,
        )
        V_slices = rearrange(
            V,
            "... sequence_length (num_heads hd_v) -> ... num_heads sequence_length hd_v",
            num_heads=self.num_heads,
        )
        sequence_length = Q.shape[-2]
        mask = torch.tril(
            torch.ones(
                sequence_length, sequence_length, device=Q.device, dtype=torch.bool
            )
        )
        # FLOPs by matmul: 2 * sequence_length^2 * (d_k+d_v) =4 * sequence_length^2 * d_model
        if self.rope is not None:
            token_positions = torch.arange(
                sequence_length, device=Q.device, dtype=torch.long
            )
            Q_slices_roped = self.rope(Q_slices, token_positions)
            K_slices_roped = self.rope(K_slices, token_positions)
            head = scaled_dot_product_attention(
                Q_slices_roped, K_slices_roped, V_slices, mask
            )
        else:
            head = scaled_dot_product_attention(Q_slices, K_slices, V_slices, mask)
        result = rearrange(
            head,
            "... num_heads sequence_length hd_v -> ... sequence_length (num_heads hd_v)",
        )
        return result

    def forward(
        self, in_features: Float[Tensor, " ... sequence_length d_model"]
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        # FLOPs by matmul(Linear) per sequence: 8 * d_model * d_model * sequence_length
        return self.output_proj(
            self.multihead(
                self.q_proj(in_features),
                self.k_proj(in_features),
                self.v_proj(in_features),
            )
        )


# if __name__ == "__main__":

#     d_model = 4
#     num_heads = 2
#     sequence_length = 3
#     q_proj_weight = torch.tensor(
#         [[1, 2, 3, 4], [3, 4, 2, 1], [2, 1, 3, 4], [1, 3, 2, 4]], dtype=torch.float
#     )
#     k_proj_weight = torch.tensor(
#         [[1, 3, 2, 4], [2, 4, 1, 3], [3, 4, 2, 1], [2, 1, 3, 4]], dtype=torch.float
#     )
#     v_proj_weight = torch.tensor(
#         [[2, 3, 1, 4], [4, 1, 2, 3], [2, 4, 1, 3], [3, 4, 2, 1]], dtype=torch.float
#     )
#     o_proj_weight = torch.tensor(
#         [[2, 4, 3, 1], [1, 3, 4, 2], [4, 1, 2, 3], [2, 4, 1, 3]], dtype=torch.float
#     )
#     x = torch.tensor([[1, 1, 2, 3], [3, 3, 2, 1], [2, 2, 1, 3]], dtype=torch.float)
#     multihead_self_attention = Multihead_Self_Attention(
#         d_model, num_heads, max_seq_len=3, theta=1
#     )
#     multihead_self_attention.W_Q.weight.data = q_proj_weight
#     multihead_self_attention.W_K.weight.data = k_proj_weight
#     multihead_self_attention.W_V.weight.data = v_proj_weight
#     multihead_self_attention.W_O.weight.data = o_proj_weight

#     print(multihead_self_attention(x))
