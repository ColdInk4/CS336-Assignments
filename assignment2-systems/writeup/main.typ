#set document(
  title: "CS336 Assignment 2: Systems and Parallelism",
  author: "",
)
#set page(paper: "us-letter", margin: (x: 0.85in, y: 0.8in))
#set text(font: ("New Computer Modern", "Noto Serif CJK SC", "Noto Sans CJK SC"), size: 10.5pt)
#show raw: set text(font: ("DejaVu Sans Mono", "Noto Sans Mono CJK SC"), size: 9pt)
#set par(justify: true, leading: 0.58em)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 18pt, weight: "bold")[CS336 Assignment 2: Systems and Parallelism]

  #v(0.3em)
  #text(fill: rgb("#4b5563"))[Writeup]
]

#v(1.2em)
#outline(title: [Contents])
#pagebreak()

#include "problems/P1_benchmarking_script.typ"
#include "problems/P2_nsys_profile.typ"
#include "problems/P3_mixed_precision_accumulation.typ"
#include "problems/P4_benchmarking_mixed_precision.typ"
#include "problems/P5_memory_profiling.typ"
#include "problems/P6_gradient_checkpointing.typ"
#include "problems/P7_pytorch_attention.typ"
#include "problems/P8_torch_compile.typ"
#include "problems/P9_flash_benchmarking.typ"
#include "problems/P10_distributed_communication_single_node.typ"
#include "problems/P11_naive_ddp_benchmarking.typ"
#include "problems/P12_minimal_ddp_flat_benchmarking.typ"
#include "problems/P13_ddp_overlap_individual_parameters_benchmarking.typ"
#include "problems/P14_optimizer_state_sharding_accounting.typ"
#include "problems/P15_fsdp_accounting.typ"
#include "problems/P16_alternate_ring_all_reduce.typ"
#include "problems/P17_data_parallel_calcs.typ"
#include "problems/P18_fsdp_calcs.typ"
#include "problems/P19_tp_calcs.typ"
#include "problems/P20_fsdp_tp_calcs.typ"
#include "problems/P21_leaderboard.typ"
