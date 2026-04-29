#import "../template.typ": problem, deliverable, note, section-label, subhead, prompt-box, answer-box, plot

#problem("pre_norm_ablation", "Implement post-norm and train (0.5 B200 hrs)", "1 point")[
#prompt-box[
#section-label[Prompt]

Modify your pre-norm Transformer implementation into a post-norm one. Train with the post-norm model and see what happens.

#deliverable[A learning curve for a post-norm Transformer, compared to the pre-norm one.]

]

#answer-box[
#section-label[Answer]

#plot("assets/P13_post_norm_01_train.png")
#plot("assets/P13_post_norm_02_valid.png")

我将默认的 pre-norm Transformer 改成 post-norm Transformer，并在相同训练设置下比较。两者使用相同的 batch size、learning rate schedule 和 token budget。

从训练曲线看，两者都能稳定收敛，没有出现明显发散。不过 pre-norm 的 loss 持续略低于 post-norm。到 10k steps 时，pre-norm 的 train loss 约为 1.273，valid loss 约为 1.293；post-norm 的 train loss 约为 1.322，valid loss 约为 1.323。

因此，在本实验的小模型和 TinyStories 设置下，post-norm 仍然可以训练，但表现略差于 pre-norm。这说明 norm 的位置会影响优化效果。pre-norm 可能让每个 sublayer 接收到尺度更稳定的输入，并改善梯度传播；不过由于模型较小、训练任务较简单，这里的差距没有表现为严重不稳定或发散。
]
]
