#import "../template.typ": problem, deliverable, note, section-label, subhead, prompt-box, answer-box, plot, answer

#problem("optimizer_state_sharding_accounting", "Optimizer State Sharding Accounting", "5 points")[
#prompt-box[
#section-label[Prompt]

(a) Profile peak memory usage when training language models with and without optimizer state sharding, using 1 node, 2 GPUs, and the XL model size. Report memory after model initialization, directly before the optimizer step, and directly after the optimizer step.

#deliverable[A 2-3 sentence response with peak memory usage results and a breakdown by model and optimizer components.]

(b) How does optimizer state sharding affect training speed? Measure the time per iteration with and without optimizer state sharding for the same standard configuration.

#deliverable[A 2-3 sentence response with your timings.]

(c) How does this optimizer state sharding approach differ from ZeRO stage 1, especially with respect to memory usage and communication?

#deliverable[A 2-3 sentence summary of the differences.]
]

#answer()
]
