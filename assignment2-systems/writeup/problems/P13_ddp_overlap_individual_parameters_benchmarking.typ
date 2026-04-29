#import "../template.typ": problem, deliverable, note, section-label, subhead, prompt-box, answer-box, plot, answer

#problem("ddp_overlap_individual_parameters_benchmarking", "DDP Overlapping Individual Parameters Benchmarking", "1 point")[
#prompt-box[
#section-label[Prompt]

(a) Benchmark DDP when overlapping backward computation with communication of individual parameter gradients, and compare it to your earlier DDP implementations.

#deliverable[The measured time per training iteration when overlapping backward computation with communication, with 1-2 sentences comparing the results.]

(b) Instrument the benchmark with Nsight Systems using the 1 node, 2 GPU, XL model setup.

#deliverable[Two screenshots, one from the initial DDP implementation and one from the overlapping implementation, visually showing whether communication overlaps with the backward pass.]
]

#answer()
]
