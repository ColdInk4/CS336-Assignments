#import "../template.typ": problem, deliverable, note, section-label, subhead, prompt-box, answer-box, plot, answer

#problem("fsdp_calcs", "Fully Sharded Data Parallel Calculations", "3 points")[
#prompt-box[
#section-label[Prompt]

(a) How many FLOPs are required to compute the backward pass with `N_FSDP` FSDP? What about the forward pass?

#deliverable[Two answers in terms of `B`, `D`, `D_FF`, and `N_FSDP`, along with two one-sentence justifications.]

(b) How much communication time is required in the backward pass with `N_FSDP` FSDP? What about the forward pass?

#deliverable[Two answers in terms of a subset of `B`, `D`, `D_FF`, `N_FSDP`, and `W`, along with two one-sentence justifications.]

(c) Fixing the other parameters, how large can `N_FSDP` become before the backward pass is communication bottlenecked? What about the forward pass?

#deliverable[Two inequalities with `N_FSDP` on one side, and an expression in terms of a subset of `B`, `D`, `D_FF`, `C`, and `W` on the other, along with two one-sentence justifications.]
]

#answer()
]
