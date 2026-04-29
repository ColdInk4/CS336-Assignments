## Problem (data_parallel_calcs): Data Parallel Calculations (3 points)

### Prompt

(a) How many FLOPs are required to compute the backward pass with `N_DP` data parallelism?

> Deliverable: An answer in terms of `B`, `D`, `D_FF`, and `N_DP`, along with a one-sentence justification.

(b) How much communication time is required in the backward pass with `N_DP` data parallelism?

> Deliverable: An answer in terms of a subset of `B`, `D`, `D_FF`, `N_DP`, and `W`, along with a one-sentence justification.

(c) Fixing the other parameters, how large can `N_DP` become before the backward pass is communication bottlenecked?

> Deliverable: An inequality with `N_DP` on one side, and an expression in terms of a subset of `B`, `D`, `D_FF`, and `W` on the other.

### Answer

