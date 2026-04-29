#import "../template.typ": problem, deliverable, note, section-label, subhead, prompt-box, answer-box, plot, answer

#problem("benchmarking_mixed_precision", "Benchmarking Mixed Precision", "2 points")[
#prompt-box[
#section-label[Prompt]

(a) For the model shown in the handout under FP16 autocasting, identify the data types of each listed component.

#deliverable[The data types for each of the components listed in the handout.]

(b) FP16 mixed-precision autocasting treats layer normalization differently from feed-forward layers. What parts of layer normalization are sensitive to precision? If using BF16 instead of FP16, do we still need to treat layer normalization differently, and why?

#deliverable[A 2-3 sentence response.]

(c) Run the benchmark using mixed precision with BF16 and compare against full precision for each language model size described in the handout.

#deliverable[A 2-3 sentence response with your timings and commentary.]
]

#answer()
]
