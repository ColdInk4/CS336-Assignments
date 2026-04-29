#import "../template.typ": problem, deliverable, note, section-label, subhead, prompt-box, answer-box, plot

#problem("no_pos_emb", "Implement NoPE (0.5 B200 hrs)", "1 point")[
#prompt-box[
#section-label[Prompt]

Modify your Transformer implementation with RoPE to remove the position embedding information entirely, and see what happens.

#deliverable[A learning curve comparing the performance of RoPE and NoPE.]

]

#answer-box[
#section-label[Answer]

#plot("assets/P14_nope_01_train.png")
#plot("assets/P14_nope_02_valid.png")

我比较了使用 RoPE 的 baseline 和完全移除位置编码的 NoPE。两者使用相同的训练设置和 learning rate schedule，只是在 NoPE 中不再对 Q/K 应用 RoPE，causal mask 仍然保留。

从 learning curves 来看，NoPE 仍然可以正常训练，validation loss 持续下降，说明 decoder-only Transformer 确实可以在没有显式位置编码的情况下从 causal mask 和数据分布中学习到一部分位置信息。不过，NoPE 的 train loss 和 validation loss 都明显高于 RoPE，收敛也更慢。RoPE 最终取得了更低的 validation loss，说明显式的相对位置信息对语言建模性能有明显帮助。

NoPE 的 tokens/s 略高，因为省去了 RoPE 计算，但这个计算收益没有弥补 loss 上的性能下降。因此，在本实验设置下，RoPE 比 NoPE 更有效。
]
]
