## Problem (nsys_profile): Nsight Systems Profiling (5 points)

### Prompt

(a) What is the total time spent on your forward pass? Does it match what you measured with your benchmarking script?

> Deliverable: A 1-2 sentence response.

(b) What CUDA kernel takes the most cumulative GPU time during the forward pass? How many times is this kernel invoked during a single forward pass? Is it the same kernel you expected from FLOP accounting?

> Deliverable: A 1-2 sentence response.

(c) Besides matrix multiplications, what other kernels take non-trivial runtime?

> Deliverable: A 1-2 sentence response.

(d) Profile one complete training step with AdamW. Compared to inference-only profiling, how does the fraction of time spent on matrix multiplication change? How about other kernels?

> Deliverable: A 1-2 sentence response.

(e) Compare the runtime of softmax versus the matrix multiplication operations inside self-attention during a forward pass. How does the difference compare to the FLOPs difference?

> Deliverable: A 1-2 sentence response.

### Answer

