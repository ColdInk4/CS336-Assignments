import torch
import os
import typing


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    map_location=None,
):
    checkpoint = {
        "iteration": iteration,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    map_location=None,
) -> int:
    checkpoint = torch.load(src, map_location=map_location, weights_only=True)

    iteration = checkpoint["iteration"]
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return iteration


def load_model_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    map_location=None,
):
    checkpoint = torch.load(src, map_location=map_location, weights_only=True)
    model.load_state_dict(checkpoint["model"])
