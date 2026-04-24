# FLOPs by matmul per attention call: 2 * queries * d_k * keys + 2 * queries * d_v * keys

from torch import Tensor
from jaxtyping import Float, Bool
from cs336_basics.Transformer import softmax
from einops import einsum
from math import sqrt


def scaled_dot_product_attention(
    Q: Float[Tensor, "... queries d_k"],
    K: Float[Tensor, "... keys d_k"],
    V: Float[Tensor, "... keys d_v"],
    mask: Bool[Tensor, "... queries keys"] | None = None,
) -> Float[Tensor, "... queries d_v"]:
    d_k = Q.shape[-1]
    # FLOPs by matmul: 2 * queries * d_k * keys
    pre_softmax_values: Float[Tensor, "... queries keys"] = einsum(
        Q,
        K,
        "... queries d_k, ... keys d_k -> ... queries keys",
    ) / sqrt(d_k)
    if mask is not None:
        pre_softmax_values = pre_softmax_values.masked_fill(~mask, -float("inf"))
    softmax_values = softmax(pre_softmax_values, dim=-1)
    # FLOPs by matmul: 2 * queries * d_v * keys
    result = einsum(
        softmax_values,
        V,
        "... queries keys, ... keys d_v -> ... queries d_v",
    )
    return result


# queries = 4
# d_k = 8
# keys = 2
# d_v = 4
# Q = torch.rand(2, queries, d_k)
# K = torch.rand(2, keys, d_k)
# V = torch.rand(2, keys, d_v)
# mask = torch.tensor([[True, True], [True, False], [False, True], [False, False]])
# print(scaled_dot_product_attention(Q, K, V, mask))
