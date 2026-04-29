## Problem (memory_profiling): Memory Profiling (4 points)

### Prompt

(a) Profile the XL model with the PyTorch memory profiler for inference-only and for a full training step.

> Deliverable: Two images of the "Active memory timeline" from memory_viz.

(b) For each context length, what is the peak memory usage during a forward pass, and what is the peak memory usage during a full training step?

> Deliverable: A table with two numbers per context length.

(c) Find the peak memory usage of the XL model using mixed precision for both a forward pass and a full training step. Does mixed precision significantly affect memory usage?

> Deliverable: A 2-3 sentence response.

(d) For the XL model and reference hyperparameters, what is the size of a tensor of activations with shape `[batch_size, sequence_length, d_model]`?

> Deliverable: A 1-2 sentence response with your derivation.

(e) Inspect the largest allocations in the Active Memory Timeline. What is the size of the largest allocations, and what tensors do they correspond to?

> Deliverable: A 1-2 sentence response.

(f) Use Nsight Systems memory profiling to inspect how much memory different TransformerBlock modules take. Does the result match what you expect?

> Deliverable: Screenshots from Nsight Systems and a 1-2 paragraph response.

### Answer

