## Problem (minimal_ddp_flat_benchmarking): Minimal DDP with Flat Gradients Benchmarking (2 points)

### Prompt

Benchmark the minimal DDP implementation that uses a single flattened gradient buffer and compare it with the naive implementation that issues one all-reduce per parameter.

> Deliverable: The measured time per training iteration and time spent communicating gradients for both implementations, with a short comparison.

### Answer

