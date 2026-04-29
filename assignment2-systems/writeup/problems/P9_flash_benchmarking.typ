#import "../template.typ": problem, deliverable, note, section-label, subhead, prompt-box, answer-box, plot, answer

#problem("flash_benchmarking", "FlashAttention-2 Benchmarking", "5 points")[
#prompt-box[
#section-label[Prompt]

Benchmark your FlashAttention-2 implementation using `triton.testing.do_bench` and compare it against regular PyTorch attention. Use a single B200, batch size 1, causal masking, sequence lengths from 128 to 65536 in powers of two, embedding dimensions from 16 to 128 in powers of two, and both `torch.bfloat16` and `torch.float32`.

#deliverable[A table comparing FlashAttention-2 with the PyTorch implementation, reporting forward, backward, and end-to-end forward-backward latencies.]
]

#answer()
]
