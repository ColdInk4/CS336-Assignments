#import "../template.typ": problem, deliverable, note, section-label, subhead, prompt-box, answer-box, plot, answer

#problem("fsdp_tp_calcs", "2D Parallelism Calculations", "6 points")[
#prompt-box[
#section-label[Prompt]

(a) How many FLOPs are required to compute the forward pass with `N_FSDP` FSDP and `N_TP` tensor parallelism?

#deliverable[An answer in terms of `B`, `D`, `D_FF`, `N_FSDP`, and `N_TP`, along with a one-sentence justification.]

(b) How much communication time is required in the forward pass with `N_FSDP` FSDP and `N_TP` tensor parallelism? Assume the communication along the FSDP and TP axes can be overlapped.

#deliverable[An answer in terms of a subset of `B`, `D`, `D_FF`, `N_FSDP`, `N_TP`, and `W`, expressed as a max between the two overlappable collective costs, along with a one-sentence justification.]

(c) Under the optimal setting of `N_TP` and `N_FSDP`, how large can `N = N_TP N_FSDP` become before the forward pass is communication bottlenecked?

#deliverable[An inequality with `N` on one side, and an expression in terms of a subset of `B`, `D`, `D_FF`, `C`, and `W` on the other, along with a few sentences and equations as justification.]

(d) Suppose the FSDP-axis and TP-axis collectives cannot be overlapped because they share the same network resources. Under the optimal setting of `N_TP` and `N_FSDP`, how large can `N = N_TP N_FSDP` become before the forward pass is communication bottlenecked?

#deliverable[An inequality with `N` on one side, and an expression in terms of a subset of `B`, `D`, `D_FF`, `C`, and `W` on the other, along with a few sentences and equations as justification.]
]

#answer()
]
