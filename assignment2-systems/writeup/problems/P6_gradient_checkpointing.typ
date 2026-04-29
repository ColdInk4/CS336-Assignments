#import "../template.typ": problem, deliverable, note, section-label, subhead, prompt-box, answer-box, plot, answer

#problem("gradient_checkpointing", "Memory-Optimal Gradient Checkpointing", "4 points")[
#prompt-box[
#section-label[Prompt]

(a) What checkpointing strategy minimizes peak activation memory if compute cost is ignored?

#deliverable[A 3-5 sentence description of the strategy, its asymptotic peak memory and compute, plus a short code sketch.]

(b) For the XL model with batch size 4 and sequence length 2048, if you can run only one step of recomputation with no nested checkpoint calls, choose a checkpointing strategy and validate it with measured peak memory.

#deliverable[A 3-5 sentence description of your reasoning along with the measured peak memory for your strategy.]
]

#answer()
]
