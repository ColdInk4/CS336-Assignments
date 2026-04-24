import torch
from torch import Tensor
from jaxtyping import Float


def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    max_values, _ = in_features.max(dim=dim, keepdim=True)
    exp_shifted = torch.exp(in_features - max_values)
    softmax_values = exp_shifted / exp_shifted.sum(dim=dim, keepdim=True)

    return softmax_values


# a = torch.tensor([[1.0, 2.0, 2.0], [1.0, 2.0, 2.0]])
# dim = 1

# print(softmax(a, 0), softmax(a, 1))
