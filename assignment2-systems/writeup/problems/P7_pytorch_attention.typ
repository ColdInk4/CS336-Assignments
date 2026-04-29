#import "../template.typ": problem, deliverable, note, section-label, subhead, prompt-box, answer-box, plot, answer

#problem("pytorch_attention", "PyTorch Attention Benchmarking", "2 points")[
#prompt-box[
#section-label[Prompt]

Benchmark your attention implementation at different scales with batch size 8, no multihead dimension, head dimensions `[16, 32, 64, 128]`, and sequence lengths `[256, 1024, 4096, 8192, 16384]`. Compare forward/backward timings and saved memory. How does the memory saved for backward change with sequence length? What would you do to eliminate this memory usage?

#deliverable[A table with your timings, your calculations for memory usage, and a 1-2 paragraph response.]
]

#answer()
]
