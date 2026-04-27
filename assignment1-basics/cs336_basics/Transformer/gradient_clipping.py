from collections.abc import Iterable
from torch import nn
import torch


def gradient_clipping(parameters: Iterable[nn.Parameter], max_l2_norm: float) -> None:
    eps = 1e-6
    parameters = list(parameters)
    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]

    if not grads:
        return

    grad_squared_sum = (grads[0] ** 2).sum()

    for grad in grads[1:]:
        grad_squared_sum += (grad**2).sum()

    global_l2_norm = grad_squared_sum.sqrt()
    scale = max_l2_norm / (global_l2_norm + eps)
    if global_l2_norm > max_l2_norm:
        for grad in grads:
            grad.mul_(scale)
