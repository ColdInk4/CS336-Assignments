## Problem (learning_rate_tuning): Tuning the learning rate (1 point)

### Prompt

As we will see, one of the hyperparameters that affects training the most is the learning rate. Let’s see that in practice in our toy example.

Run the SGD example above with three other values for the learning rate, `1e1`, `1e2`, and `1e3`, for just 10 training iterations. What happens with the loss for each of these learning rates? Does it decay faster, slower, or does it diverge, that is, increase over the course of training?

> Deliverable: A one-to-two sentence response with the behaviors you observed.

### Answer

当 lr = 1e1 时，loss 稳定下降；当 lr = 1e2 时，loss 下降得更快并迅速接近 0。 当 lr = 1e3 时，loss 在 10 次迭代中快速增大，表现出发散趋势。
