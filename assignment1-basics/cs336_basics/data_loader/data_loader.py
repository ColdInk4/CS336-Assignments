import numpy.typing as npt
import torch
from jaxtyping import Int
from torch import Tensor
import random


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[
    Int[Tensor, "batch_size context_length"], Int[Tensor, "batch_size context_length"]
]:
    input_sequences = torch.zeros(
        batch_size, context_length, device=device, dtype=torch.int64
    )
    next_token_targets = torch.zeros(
        batch_size, context_length, device=device, dtype=torch.int64
    )
    for i in range(batch_size):
        start = random.randint(0, dataset.size - context_length - 1)
        input_sequences[i] = torch.tensor(
            dataset[start : start + context_length], device=device
        )
        next_token_targets[i] = torch.tensor(
            dataset[start + 1 : start + context_length + 1], device=device
        )
    return (input_sequences, next_token_targets)
