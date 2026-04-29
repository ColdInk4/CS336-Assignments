#import "../template.typ": problem, deliverable, note, section-label, subhead, prompt-box, answer-box, plot, answer

#problem("nsys_profile", "Nsight Systems Profiling", "5 points")[
#prompt-box[
#section-label[Prompt]

Profile the forward pass, backward pass, and optimizer step with Nsight Systems using two model sizes and three power-of-two context lengths larger than 128.

(a) What is the total time spent on your forward pass? Does it match what you measured with your benchmarking script?

#deliverable[A 1-2 sentence response.]

(b) What CUDA kernel takes the most cumulative GPU time during the forward pass? How many times is this kernel invoked during a single forward pass? Is it the same kernel you expected from FLOP accounting?

#deliverable[A 1-2 sentence response.]

(c) Besides matrix multiplications, what other kernels take non-trivial runtime?

#deliverable[A 1-2 sentence response.]

(d) Profile one complete training step with AdamW. Compared to inference-only profiling, how does the fraction of time spent on matrix multiplication change? How about other kernels?

#deliverable[A 1-2 sentence response.]

(e) Compare the runtime of softmax versus the matrix multiplication operations inside self-attention during a forward pass. How does the difference compare to the FLOPs difference?

#deliverable[A 1-2 sentence response.]
]

#answer()
]
