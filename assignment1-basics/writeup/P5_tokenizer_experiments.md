## Problem (tokenizer_experiments): Experiments with tokenizers (4 points)

### Prompt

(a) Sample 10 documents from TinyStories and OpenWebText. Using your previously trained TinyStories and OpenWebText tokenizers (10K and 32K vocabulary sizes, respectively), encode these sampled documents into integer ids. What is each tokenizer's compression ratio, measured in bytes per token?

> Deliverable: A one-to-two sentence response.

(b) What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer? Compare the compression ratio and/or qualitatively describe what happens.

> Deliverable: A one-to-two sentence response.

(c) Estimate the throughput of your tokenizer, for example in bytes per second. About how long would it take to tokenize the Pile dataset (825 GB of text)?

> Deliverable: A one-to-two sentence response.

(d) Using your TinyStories and OpenWebText tokenizers, encode the corresponding training and development datasets into sequences of integer token ids. We recommend storing the token ids as a NumPy array with dtype `uint16`. Why is `uint16` an appropriate choice?

> Deliverable: A one-to-two sentence response.

### Answer

a. 我在 TinyStories 和 OpenWebText 中各随机采样了 10 个文档。TinyStories tokenizer 在 TinyStories 样本上的压缩率约为 4.13 bytes/token，OpenWebText tokenizer 在 OpenWebText 样本上的压缩率约为 4.46 bytes/token；bytes/token 越高，表示每个 token 承载的原始字节越多，压缩效果越强。
b. 当我用 TinyStories tokenizer 去编码 OpenWebText 样本时，压缩率从 OpenWebText tokenizer 的约 4.46 bytes/token 降到了约 3.27 bytes/token。这说明 OpenWebText 文本会被切得更碎、需要更多 token，原因大概率是 TinyStories tokenizer 的词表更小，而且训练语料和 OpenWebText 的领域分布不匹配。
c. 我测得 tokenizer 的吞吐大约是 4 MB/s 左右。按这个速度粗略估算，tokenize 大约 825GB 的 The Pile 需要约 59-60 小时，实际时间还会受到硬件、I/O 和实现细节影响。
d. uint16 的取值范围为 0 至 65,535，足以覆盖本项目中 10,000 (10K) 和 32,000 (32K) 的词表大小；同时相比于 uint32 或默认的 int64，它能在保证不溢出的前提下将内存和存储开销减半，显著提高大规模数据集的处理效率。