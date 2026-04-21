## Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points)

### Prompt

(a) Train a byte-level BPE tokenizer on the OpenWebText dataset, using a maximum vocabulary size of 32,000. Serialize the resulting vocabulary and merges to disk for further inspection. What is the longest token in the vocabulary? Does it make sense?

> Resource requirements: ≤ 12 hours (no GPUs), ≤ 100 GB RAM

> Deliverable: A one-to-two sentence response.

(b) Compare and contrast the tokenizer that you get training on TinyStories versus OpenWebText.

> Deliverable: A one-to-two sentence response.

### Answer

a. 在 OpenWebText 上训练得到的 32,000 词表中，最长 token 是
  b'\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc
  3\x83\xc3\x82\xc3\x83\xc3\x82'，解码后对应一串 ÃÂÃÂ...。这基本可以解释为网页语料中的编码污染或乱码，放在 OpenWebText 这种抓取语料里是合理的。
b. 和 TinyStories 相比，OpenWebText 训练出来的 tokenizer 明显更“脏”也更多样，除了自然语言里的常见词片段，还学到了乱码、异常标点和网页文本噪声；而 TinyStories 的词表更干净，更偏向常见叙事文本中的高频词和带前导空格的英文词片段。