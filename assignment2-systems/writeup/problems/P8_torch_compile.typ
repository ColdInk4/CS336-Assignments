#import "../template.typ": problem, deliverable, note, section-label, subhead, prompt-box, answer-box, plot, answer

#problem("torch_compile", "Torch Compile", "2 points")[
#prompt-box[
#section-label[Prompt]

(a) Extend your attention benchmark to include a compiled version of your PyTorch attention implementation, using the same configuration as the PyTorch attention problem.

#deliverable[A table comparing forward and backward pass timings for compiled and uncompiled attention.]

(b) Compile the entire Transformer model in your end-to-end benchmarking script. How does forward pass performance change? What about combined forward and backward performance?

#deliverable[A table comparing your vanilla and compiled Transformer model.]
]

#answer()
]
