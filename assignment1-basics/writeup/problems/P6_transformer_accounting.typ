#import "../template.typ": problem, deliverable, note, section-label, subhead, prompt-box, answer-box, plot

#problem("transformer_accounting", "Transformer LM resource accounting", "5 points")[
#prompt-box[
#section-label[Prompt]

(a) Consider a GPT-2 XL-sized model using our assignment architecture, which has the following configuration:

- `vocab_size: 50,257`
- `context_length: 1,024`
- `num_layers: 48`
- `d_model: 1,600`
- `num_heads: 25`
- `d_ff: 4,288` (the nearest multiple of 64 to `8 / 3 × 1,600`)

Suppose we constructed our model using this configuration. How many trainable parameters would our model have? Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model?

#deliverable[A one-to-two sentence response.]

(b) Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped model. How many FLOPs do these matrix multiplies require in total? Assume that your input sequence has `context_length` tokens.

#deliverable[A list of matrix multiplies, with descriptions, and the total number of FLOPs required.]

(c) Based on your analysis above, which parts of the model require the most FLOPs?

#deliverable[A one-to-two sentence response.]

(d) Repeat your analysis with GPT-2 small (`12` layers, `768 d_model`, `12` heads), GPT-2 medium (`24` layers, `1024 d_model`, `16` heads), and GPT-2 large (`36` layers, `1280 d_model`, `20` heads). As the model size increases, which parts of the Transformer LM take up proportionally more or less of the total FLOPs?

#deliverable[For each model, provide a breakdown of model components and their associated FLOPs as a proportion of the total FLOPs required for a forward pass. In addition, provide a one-to-two sentence description of how varying the model size changes the proportional FLOPs of each component.]

(e) Take GPT-2 XL and increase the context length to `16,384`. How does the total FLOPs for one forward pass change? How does the relative contribution of FLOPs of the model components change?

#deliverable[A one-to-two sentence response.]

]

#answer-box[
#section-label[Answer]

a. 
#raw("trainable parameters = 1640452800", block: true)
single-precision floating point 是 32 bit
1640452800 \* 32bit = 52494489600bit = 6561811200 B = 6.56GB

b.
只统计 matrix multiplies
#raw("total FLOPs: 3,516,769,894,400\ntransformer_blocks(95.32%): 3,352,087,756,800\n|- Multihead_Self_Attention(37.78%): 1,328,755,507,200\n  |- q_proj(7.16%): 251,658,240,000\n  |- k_proj(7.16%): 251,658,240,000\n  |- v_proj(7.16%): 251,658,240,000\n  |- output_proj(7.16%): 251,658,240,000\n  |- QK^T(4.58%): 161,061,273,600\n  |- softmax(QK^T)V(4.58%): 161,061,273,600\n|- FFN(57.53%): 2,023,332,249,600\n  |- w1(19.18%): 674,444,083,200\n  |- w2(19.18%): 674,444,083,200\n  |- w3(19.18%): 674,444,083,200\nlm_head(4.68%): 164,682,137,600", block: true)

c. 在 GPT-2 XL、L=1024 下，FFN/SwiGLU 是最大的 FLOPs 来源，其次是 multi-head attention，lm\_head 占比较小。

d.
#raw("GPT-2-small(trainable_parameters: 162,148,608): \ntotal FLOPs: 291,648,307,200\ntransformer_blocks(72.90%): 212,600,881,152\n|- Multihead_Self_Attention(33.13%): 96,636,764,160\n  |- q_proj(4.97%): 14,495,514,624\n  |- k_proj(4.97%): 14,495,514,624\n  |- v_proj(4.97%): 14,495,514,624\n  |- output_proj(4.97%): 14,495,514,624\n  |- QK^T(6.63%): 19,327,352,832\n  |- softmax(QK^T)V(6.63%): 19,327,352,832\n|- FFN(39.76%): 115,964,116,992\n  |- w1(13.25%): 38,654,705,664\n  |- w2(13.25%): 38,654,705,664\n  |- w3(13.25%): 38,654,705,664\nlm_head(27.10%): 79,047,426,048\n\nGPT-2-medium(trainable_parameters: 406,539,264): \ntotal FLOPs: 830,172,299,264\ntransformer_blocks(87.30%): 724,775,731,200\n|- Multihead_Self_Attention(37.25%): 309,237,645,312\n  |- q_proj(6.21%): 51,539,607,552\n  |- k_proj(6.21%): 51,539,607,552\n  |- v_proj(6.21%): 51,539,607,552\n  |- output_proj(6.21%): 51,539,607,552\n  |- QK^T(6.21%): 51,539,607,552\n  |- softmax(QK^T)V(6.21%): 51,539,607,552\n|- FFN(50.05%): 415,538,085,888\n  |- w1(16.68%): 138,512,695,296\n  |- w2(16.68%): 138,512,695,296\n  |- w3(16.68%): 138,512,695,296\nlm_head(12.70%): 105,396,568,064\n\nGPT-2-large(trainable_parameters: 833,591,040):\ntotal FLOPs: 1,768,530,903,040\ntransformer_blocks(92.55%): 1,636,785,192,960\n|- Multihead_Self_Attention(38.25%): 676,457,349,120\n  |- q_proj(6.83%): 120,795,955,200\n  |- k_proj(6.83%): 120,795,955,200\n  |- v_proj(6.83%): 120,795,955,200\n  |- output_proj(6.83%): 120,795,955,200\n  |- QK^T(5.46%): 96,636,764,160\n  |- softmax(QK^T)V(5.46%): 96,636,764,160\n|- FFN(54.30%): 960,327,843,840\n  |- w1(18.10%): 320,109,281,280\n  |- w2(18.10%): 320,109,281,280\n  |- w3(18.10%): 320,109,281,280\nlm_head(7.45%): 131,745,710,080", block: true)

随着模型规模在固定 context_length 下增大，Transformer blocks 的 FLOPs 占比上升，lm_head 占比下降。块内主要增长的是 FFN 和 attention 中的 projection matmuls，而真正的序列二次项 QK^T 与 softmax(QK^T)V 的相对占比反而下降。

e.
#raw("context_length: 16,384\ntotal FLOPs: 133,577,729,638,400\ntransformer_blocks(98.03%): 130,942,815,436,800\n|- Multihead_Self_Attention(73.79%): 98,569,499,443,200\n  |- q_proj(3.01%): 4,026,531,840,000\n  |- k_proj(3.01%): 4,026,531,840,000\n  |- v_proj(3.01%): 4,026,531,840,000\n  |- output_proj(3.01%): 4,026,531,840,000\n  |- QK^T(30.87%): 41,231,686,041,600\n  |- softmax(QK^T)V(30.87%): 41,231,686,041,600\n|- FFN(24.24%): 32,373,315,993,600\n  |- w1(8.08%): 10,791,105,331,200\n  |- w2(8.08%): 10,791,105,331,200\n  |- w3(8.08%): 10,791,105,331,200\nlm_head(1.97%): 2,634,914,201,600", block: true)

相比较之前的：

#raw("context_length: 1,024\ntotal FLOPs: 3,516,769,894,400\ntransformer_blocks(95.32%): 3,352,087,756,800\n|- Multihead_Self_Attention(37.78%): 1,328,755,507,200\n  |- q_proj(7.16%): 251,658,240,000\n  |- k_proj(7.16%): 251,658,240,000\n  |- v_proj(7.16%): 251,658,240,000\n  |- output_proj(7.16%): 251,658,240,000\n  |- QK^T(4.58%): 161,061,273,600\n  |- softmax(QK^T)V(4.58%): 161,061,273,600\n|- FFN(57.53%): 2,023,332,249,600\n  |- w1(19.18%): 674,444,083,200\n  |- w2(19.18%): 674,444,083,200\n  |- w3(19.18%): 674,444,083,200\nlm_head(4.68%): 164,682,137,600", block: true)

自注意力层明显占比上升，尤其是计算 QK^T 和 softmax(QK^T)V 的部分。
]
]
