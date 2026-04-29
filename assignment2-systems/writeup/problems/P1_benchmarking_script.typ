#import "../template.typ": problem, deliverable, note, section-label, subhead, prompt-box, answer-box, plot, answer

#problem("benchmarking_script", "Benchmarking Script", "4 points")[
#prompt-box[
#section-label[Prompt]

(a) Implement the benchmarking script described in the handout. This writeup records the written follow-up parts below.

(b) Time the forward pass, backward pass, and optimizer step for the model sizes described in the handout. Use 5 warmup steps and compute the average and standard deviation over 10 measurement steps.

#deliverable[A 1-2 sentence response with your timings.]

(c) Repeat the timing analysis without warmup steps. How does this affect your results, and why?

#deliverable[A 2-3 sentence response.]
]

#answer()
]
