#import "../template.typ": problem, deliverable, note, section-label, subhead, prompt-box, answer-box, plot, answer

#problem("memory_profiling", "Memory Profiling", "4 points")[
#prompt-box[
#section-label[Prompt]

(a) Profile the XL model with the PyTorch memory profiler for inference-only and for a full training step. Compare context lengths 128 and 2048 where requested in the handout.

#deliverable[Two images of the "Active memory timeline" from memory_viz, plus a 2-3 sentence response.]

(b) For each context length, what is the peak memory usage during a forward pass, and what is the peak memory usage during a full training step?

#deliverable[A table with two numbers per context length.]

(c) Find the peak memory usage of the XL model using mixed precision for both a forward pass and a full training step. Does mixed precision significantly affect memory usage?

#deliverable[A 2-3 sentence response.]

(d) For the XL model and reference hyperparameters, what is the size in MiB of a single-precision residual-stream activation tensor?

#deliverable[A 1-2 sentence response with your derivation.]

(e) Inspect the largest allocations in the Active Memory Timeline. What is the size of the largest allocations, and what tensors do they correspond to?

#deliverable[A 1-2 sentence response.]

(f) Use Nsight Systems memory profiling to inspect how much memory is saved for backward by a single TransformerBlock. Note the five largest contributing operations and compare the gradient-tensor memory with what you expect.

#deliverable[Screenshots from Nsight Systems and a 1-2 paragraph response.]
]

#answer()
]
