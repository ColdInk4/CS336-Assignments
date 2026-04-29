#import "../template.typ": problem, deliverable, note, section-label, subhead, prompt-box, answer-box, plot, answer

#problem("naive_ddp_benchmarking", "Naive DDP Benchmarking", "3 points")[
#prompt-box[
#section-label[Prompt]

Benchmark the naive DDP implementation that all-reduces parameter gradients individually across ranks. Use the single-node setup from the handout: 1 node, 2 GPUs, and the XL model size.

#deliverable[A description of your benchmarking setup, along with the measured time per training iteration and time spent communicating gradients.]
]

#answer()
]
