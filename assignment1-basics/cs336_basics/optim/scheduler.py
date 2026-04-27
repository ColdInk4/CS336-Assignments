from math import cos, pi


def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:

    if it < warmup_iters:
        lr = it / warmup_iters * max_learning_rate
    elif it > cosine_cycle_iters:
        lr = min_learning_rate
    else:
        lr = min_learning_rate + 1 / 2 * (
            (1 + cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * pi))
            * (max_learning_rate - min_learning_rate)
        )
    return lr
