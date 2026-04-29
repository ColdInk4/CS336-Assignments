## Problem (tp_calcs): Tensor Parallel Calculations (4 points)

### Prompt

(a) Given input `dy` of size `(B, D)`, write out the backward pass of the tensor parallel strategy described in the handout.

> Deliverable: A series of equations describing the backward pass in terms of `dy`, sharded weights, and intermediate gradients.

(b) How many FLOPs are required to compute the forward pass with `N_TP` tensor parallelism? What about the backward pass?

> Deliverable: Two answers in terms of `B`, `D`, `D_FF`, and `N_TP`, along with two one-sentence justifications.

(c) How much communication time is required in the forward pass with `N_TP` tensor parallelism? What about the backward pass?

> Deliverable: Two answers in terms of a subset of `B`, `D`, `D_FF`, `N_TP`, and `W`, along with two one-sentence justifications.

(d) Fixing the other parameters, how large can `N_TP` become before the backward pass is communication bottlenecked? What about the forward pass?

> Deliverable: Two inequalities with `N_TP` on one side, and an expression in terms of a subset of `B`, `D`, `D_FF`, and `W` on the other.

### Answer

