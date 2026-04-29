## Problem (fsdp_tp_calcs): 2D Parallelism Calculations (6 points)

### Prompt

(a) How many FLOPs are required to compute the forward pass with `N_FSDP` FSDP and `N_TP` tensor parallelism?

> Deliverable: An answer in terms of `B`, `D`, `D_FF`, `N_FSDP`, and `N_TP`, along with a one-sentence justification.

(b) How much communication time is required in the forward pass with `N_FSDP` FSDP and `N_TP` tensor parallelism?

> Deliverable: An answer in terms of a subset of `B`, `D`, `D_FF`, `N_FSDP`, `N_TP`, and `W`, along with a one-sentence justification.

(c) Under the optimal setting of `N_TP` and `N_FSDP`, how large can `N = N_TP N_FSDP` become before the forward pass is communication bottlenecked?

> Deliverable: An inequality with `N` on one side, and an expression in terms of a subset of `B`, `D`, `D_FF`, and `W` on the other.

(d) Suppose the FSDP-axis and TP-axis collectives cannot be overlapped because they share the same network resources. How does the communication bottleneck condition change?

> Deliverable: An inequality with `N` on one side, and an expression in terms of a subset of `B`, `D`, `D_FF`, and `W` on the other.

### Answer

