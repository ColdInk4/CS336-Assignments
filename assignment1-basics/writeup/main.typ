#set document(
  title: "CS336 Assignment 1: Basics",
  author: "",
)
#set page(paper: "us-letter", margin: (x: 0.85in, y: 0.8in))
#set text(font: ("New Computer Modern", "Noto Serif CJK SC", "Noto Sans CJK SC"), size: 10.5pt)
#show raw: set text(font: ("DejaVu Sans Mono", "Noto Sans Mono CJK SC"), size: 9pt)
#set par(justify: true, leading: 0.58em)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 18pt, weight: "bold")[CS336 Assignment 1: Basics]

  #v(0.3em)
  #text(fill: rgb("#4b5563"))[Writeup]
]

#v(1.2em)
#outline(title: [Contents])
#pagebreak()

#include "problems/P1_unicode1.typ"
#include "problems/P2_unicode2.typ"
#include "problems/P3_train_bpe_tinystories.typ"
#include "problems/P4_train_bpe_expts_owt.typ"
#include "problems/P5_tokenizer_experiments.typ"
#include "problems/P6_transformer_accounting.typ"
#include "problems/P7_learning_rate_tuning.typ"
#include "problems/P8_adamw_accounting.typ"
#include "problems/P9_learning_rate.typ"
#include "problems/P10_batch_size_experiment.typ"
#include "problems/P11_generate.typ"
#include "problems/P12_layer_norm_ablation.typ"
#include "problems/P13_pre_norm_ablation.typ"
#include "problems/P14_no_pos_emb.typ"
#include "problems/P15_swiglu_ablation.typ"
#include "problems/P16_main_experiment.typ"
#include "problems/P17_leaderboard.typ"
