from jaxtyping import Float, Int
from torch import Tensor
import torch


def cross_entropy(
    inputs: Float[Tensor, "batch_size vocab_size"], targets: Int[Tensor, "batch_size"]
) -> Float[Tensor, ""]:
    batch_size = inputs.shape[0]
    max_logits: Float[Tensor, "batch_size 1"]
    max_logits, _ = inputs.max(dim=-1, keepdim=True)
    exp_shifted: Float[Tensor, "batch_size vocab_size"]
    exp_shifted = (inputs - max_logits).exp()
    exp_shifted_sum: Float[Tensor, "batch_size 1"]
    exp_shifted_sum = exp_shifted.sum(dim=-1, keepdim=True)

    # stable CE = log(sum_j exp(logit_j - max_logit)) - (correct_logit - max_logit)
    row_idx = torch.arange(batch_size, device=inputs.device)
    correct_logits = inputs[row_idx, targets][:, None]
    loss = (exp_shifted_sum.log() - (correct_logits - max_logits)).sum() / batch_size

    return loss


# D = 3

# inputs = torch.tensor(
#     [
#         [10, 11, 12, 13],
#         [20, 21, 22, 23],
#         [30, 31, 32, 33],
#     ]
# )

# targets = torch.tensor([2, 0, 3])
# cross_entropy(inputs, targets)
# row_idx = torch.arange(D)
# print(inputs[row_idx, targets])
