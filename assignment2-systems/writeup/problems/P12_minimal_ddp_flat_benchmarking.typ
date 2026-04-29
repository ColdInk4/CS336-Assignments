#import "../template.typ": problem, deliverable, note, section-label, subhead, prompt-box, answer-box, plot, answer

#problem("minimal_ddp_flat_benchmarking", "Minimal DDP with Flat Gradients Benchmarking", "2 points")[
#prompt-box[
#section-label[Prompt]

Benchmark the minimal DDP implementation that uses a single flattened gradient buffer and compare it with the naive implementation that issues one all-reduce per parameter. Use 1 node, 2 GPUs, and the XL model size.

#deliverable[The measured time per training iteration and time spent communicating gradients for the single batched all-reduce, with 1-2 sentences comparing batched vs. individual gradient communication.]
]

#answer()
]
