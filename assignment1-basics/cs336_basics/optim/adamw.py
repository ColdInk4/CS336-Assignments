# FLOPs per step: 14 * num_parameters
from collections.abc import Callable
from typing import Optional
import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, weight_decay, eps):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0 <= betas[0] < 1 and 0 <= betas[1] < 1):
            raise ValueError(f"Invalid betas: {betas}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        if eps < 0:
            raise ValueError(f"Invalid eps: {eps}")
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            with torch.no_grad():
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]  # Get state associated with p.
                    t = state.get("t", 0)  # Get iteration number from the state, or 0.
                    m = state.get("m", torch.zeros_like(p))
                    v = state.get("v", torch.zeros_like(p))
                    grad = p.grad  # Get the gradient of loss with respect to p.
                    t = t + 1

                    # Compute adjusted 𝛼 for iteration 𝑡
                    adjusted_lr = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                    # Apply weight decay
                    # FLOPs: 2 * num_parameters
                    p -= (lr * weight_decay) * p  # Apply weight decay
                    # Update the first moment estimate
                    # FLOPs: 3 * num_parameters
                    m = beta1 * m + (1 - beta1) * grad
                    # Update the second moment estimate
                    # FLOPs: 4 * num_parameters
                    v = beta2 * v + (1 - beta2) * (grad**2)
                    # Apply moment-adjusted weight updates
                    # FLOPs: 5 * num_parameters
                    p -= adjusted_lr * m / (v.sqrt() + eps)

                    state["t"] = t
                    state["m"] = m
                    state["v"] = v
        return loss
