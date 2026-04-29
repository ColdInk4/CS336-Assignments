#import "../template.typ": problem, deliverable, note, section-label, subhead, prompt-box, answer-box, plot, answer

#problem("fsdp_accounting", "FSDP Accounting", "5 points")[
#prompt-box[
#section-label[Prompt]

(a) Based on your analysis, how much peak memory do you expect to save by using FSDP? You may ignore the size of preallocated all-gather buffers in this calculation.

#deliverable[A 2-3 sentence response with your findings.]

(b) Profile the XL model on two GPUs and inspect the all-gather of weights. Does the observed communication cost match your expectation?

#deliverable[A 2-3 sentence response with your timings, including Nsight screenshots to support your answer.]
]

#answer()
]
