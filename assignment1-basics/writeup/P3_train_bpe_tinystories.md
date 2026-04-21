## Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)

### Prompt

(a) Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary size of 10,000. Make sure to add the TinyStories `<|endoftext|>` special token to the vocabulary. Serialize the resulting vocabulary and merges to disk for further inspection. How much time and memory did training take? What is the longest token in the vocabulary? Does it make sense?

> Resource requirements: ≤ 30 minutes (no GPUs), ≤ 30 GB RAM

> **Hint:** You should be able to get under 2 minutes for BPE training using multiprocessing during pre-tokenization and the following two facts:
>
> (a) The `<|endoftext|>` token delimits documents in the data files.
>
> (b) The `<|endoftext|>` token is handled as a special case before the BPE merges are applied.

> Deliverable: A one-to-two sentence response.

(b) Profile your code. What part of the tokenizer training process takes the most time?

> Deliverable: A one-to-two sentence response.

### Answer

a. 
    1. 大约用了 17 秒；用 /usr/bin/time -v 测得峰值内存为 194724 kB，约等于 195 MB RSS。词表中最长的 token 是 b' accomplishment'，长度为 15 bytes，这很合理，因为 TinyStories 中这类带前导空格的高频英文词片段确实容易被合并成一个 token。
    2. 从 profiling 结果看，训练时间主要花在 pre-tokenization 阶段，而不是 BPE merge 主循环。也就是说，最耗时的部分是对语料分块、正则预分词和统计 pretoken 频次，而 merge 阶段相对只占很小一部分时间。