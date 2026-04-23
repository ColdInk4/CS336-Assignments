import torch
from torch import nn, Tensor
from jaxtyping import Float, Int
from math import cos, sin
from einops import rearrange


class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        cos_precomputed: Float[Tensor, "max_seq_len num_pairs"] = torch.empty(
            max_seq_len, int(d_k / 2), device=device
        )
        sin_precomputed: Float[Tensor, "max_seq_len num_pairs"] = torch.empty(
            max_seq_len, int(d_k / 2), device=device
        )
        for position in range(max_seq_len):
            for pair_idx in range(int(d_k / 2)):
                pair_number = pair_idx + 1  # from 0-based to 1-based
                angle = position / theta ** ((2 * pair_number - 2) / d_k)
                cos_precomputed[position, pair_idx] = cos(angle)
                sin_precomputed[position, pair_idx] = sin(angle)
        self.register_buffer("cos_precomputed", cos_precomputed)
        self.register_buffer("sin_precomputed", sin_precomputed)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Int[Tensor, "... seq_len"],
    ) -> Float[Tensor, "... seq_len d_k"]:
        x_pairs: Float[Tensor, "...  seq_len num_pairs pair_components"] = rearrange(
            x,
            "... seq_len (num_pairs pair_components) -> ... seq_len num_pairs pair_components",
            pair_components=2,
        )
        cos_values: Float[Tensor, "... seq_len num_pairs"] = self.cos_precomputed[
            token_positions
        ]
        sin_values: Float[Tensor, "... seq_len num_pairs"] = self.sin_precomputed[
            token_positions
        ]

        rotated_first, rotated_second = (
            x_pairs[..., 0] * cos_values - x_pairs[..., 1] * sin_values,
            x_pairs[..., 0] * sin_values + x_pairs[..., 1] * cos_values,
        )
        rotated_pairs = torch.stack([rotated_first, rotated_second], dim=-1)

        return rearrange(
            rotated_pairs,
            "... seq_len num_pairs pair_components->... seq_len (num_pairs pair_components)",
            pair_components=2,
        )
