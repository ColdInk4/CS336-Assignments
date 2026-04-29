#import "../template.typ": problem, deliverable, note, section-label, subhead, prompt-box, answer-box, plot

#problem("learning_rate_tuning", "Tuning the learning rate", "1 point")[
#prompt-box[
#section-label[Prompt]

As we will see, one of the hyperparameters that affects training the most is the learning rate. Let’s see that in practice in our toy example.

Run the SGD example above with three other values for the learning rate, `1e1`, `1e2`, and `1e3`, for just 10 training iterations. What happens with the loss for each of these learning rates? Does it decay faster, slower, or does it diverge, that is, increase over the course of training?

#deliverable[A one-to-two sentence response with the behaviors you observed.]

]

#answer-box[
#section-label[Answer]

- `lr = 1e1`: loss 稳定下降。
- `lr = 1e2`: loss 下降得更快，并迅速接近 0。
- `lr = 1e3`: loss 在 10 次迭代中快速增大，表现出发散趋势。
]
]
