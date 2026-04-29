#import "../template.typ": problem, deliverable, note, section-label, subhead, prompt-box, answer-box, plot, answer

#problem("distributed_communication_single_node", "Distributed Communication (Single Node)", "5 points")[
#prompt-box[
#section-label[Prompt]

Benchmark the runtime of all-reduce in the single-node multi-GPU setting. Vary float32 tensor sizes over 1MB, 10MB, 100MB, and 1GB, and vary the number of GPUs/processes over 2, 4, and 6.

#deliverable[Plot(s) and/or table(s) comparing the various settings, with 2-3 sentences of commentary.]
]

#answer()
]
