## Problem (transformer_accounting): Transformer LM resource accounting (5 points)

### Prompt

(a) Consider a GPT-2 XL-sized model using our assignment architecture, which has the following configuration:

- `vocab_size: 50,257`
- `context_length: 1,024`
- `num_layers: 48`
- `d_model: 1,600`
- `num_heads: 25`
- `d_ff: 4,288` (the nearest multiple of 64 to `8 / 3 × 1,600`)

Suppose we constructed our model using this configuration. How many trainable parameters would our model have? Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model?

> Deliverable: A one-to-two sentence response.

(b) Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped model. How many FLOPs do these matrix multiplies require in total? Assume that your input sequence has `context_length` tokens.

> Deliverable: A list of matrix multiplies, with descriptions, and the total number of FLOPs required.

(c) Based on your analysis above, which parts of the model require the most FLOPs?

> Deliverable: A one-to-two sentence response.

(d) Repeat your analysis with GPT-2 small (`12` layers, `768 d_model`, `12` heads), GPT-2 medium (`24` layers, `1024 d_model`, `16` heads), and GPT-2 large (`36` layers, `1280 d_model`, `20` heads). As the model size increases, which parts of the Transformer LM take up proportionally more or less of the total FLOPs?

> Deliverable: For each model, provide a breakdown of model components and their associated FLOPs as a proportion of the total FLOPs required for a forward pass. In addition, provide a one-to-two sentence description of how varying the model size changes the proportional FLOPs of each component.

(e) Take GPT-2 XL and increase the context length to `16,384`. How does the total FLOPs for one forward pass change? How does the relative contribution of FLOPs of the model components change?

> Deliverable: A one-to-two sentence response.

### Answer

a. 
```
trainable parameters = 1640452800
```
single-precision floating point 是 32 bit
1640452800 * 32bit = 52494489600bit = 6561811200 B = 6.56GB

b.
只统计 matrix multiplies
```
total FLOPs: 3,516,769,894,400
transformer_blocks(95.32%): 3,352,087,756,800
|- Multihead_Self_Attention(37.78%): 1,328,755,507,200
  |- q_proj(7.16%): 251,658,240,000
  |- k_proj(7.16%): 251,658,240,000
  |- v_proj(7.16%): 251,658,240,000
  |- output_proj(7.16%): 251,658,240,000
  |- QK^T(4.58%): 161,061,273,600
  |- softmax(QK^T)V(4.58%): 161,061,273,600
|- FFN(57.53%): 2,023,332,249,600
  |- w1(19.18%): 674,444,083,200
  |- w2(19.18%): 674,444,083,200
  |- w3(19.18%): 674,444,083,200
lm_head(4.68%): 164,682,137,600
```

c. 在 GPT-2 XL、L=1024 下，FFN/SwiGLU 是最大的 FLOPs 来源，其次是 multi-head attention，lm_head 占比较小。
d. 
```
GPT-2-small(trainable_parameters: 162,148,608): 
total FLOPs: 291,648,307,200
transformer_blocks(72.90%): 212,600,881,152
|- Multihead_Self_Attention(33.13%): 96,636,764,160
  |- q_proj(4.97%): 14,495,514,624
  |- k_proj(4.97%): 14,495,514,624
  |- v_proj(4.97%): 14,495,514,624
  |- output_proj(4.97%): 14,495,514,624
  |- QK^T(6.63%): 19,327,352,832
  |- softmax(QK^T)V(6.63%): 19,327,352,832
|- FFN(39.76%): 115,964,116,992
  |- w1(13.25%): 38,654,705,664
  |- w2(13.25%): 38,654,705,664
  |- w3(13.25%): 38,654,705,664
lm_head(27.10%): 79,047,426,048

GPT-2-medium(trainable_parameters: 406,539,264): 
total FLOPs: 830,172,299,264
transformer_blocks(87.30%): 724,775,731,200
|- Multihead_Self_Attention(37.25%): 309,237,645,312
  |- q_proj(6.21%): 51,539,607,552
  |- k_proj(6.21%): 51,539,607,552
  |- v_proj(6.21%): 51,539,607,552
  |- output_proj(6.21%): 51,539,607,552
  |- QK^T(6.21%): 51,539,607,552
  |- softmax(QK^T)V(6.21%): 51,539,607,552
|- FFN(50.05%): 415,538,085,888
  |- w1(16.68%): 138,512,695,296
  |- w2(16.68%): 138,512,695,296
  |- w3(16.68%): 138,512,695,296
lm_head(12.70%): 105,396,568,064

GPT-2-large(trainable_parameters: 833,591,040):
total FLOPs: 1,768,530,903,040
transformer_blocks(92.55%): 1,636,785,192,960
|- Multihead_Self_Attention(38.25%): 676,457,349,120
  |- q_proj(6.83%): 120,795,955,200
  |- k_proj(6.83%): 120,795,955,200
  |- v_proj(6.83%): 120,795,955,200
  |- output_proj(6.83%): 120,795,955,200
  |- QK^T(5.46%): 96,636,764,160
  |- softmax(QK^T)V(5.46%): 96,636,764,160
|- FFN(54.30%): 960,327,843,840
  |- w1(18.10%): 320,109,281,280
  |- w2(18.10%): 320,109,281,280
  |- w3(18.10%): 320,109,281,280
lm_head(7.45%): 131,745,710,080

随着模型规模在固定 context_length 下增大，Transformer blocks 的 FLOPs 占比上升，lm_head 占比下降。块内主要增长的是 FFN 和 attention 中的 projection matmuls，而真正的序列二次项 QK^T 与 softmax(QK^T)V 的相对占比反而下降。

e.
```(context_length: 16,384)
total FLOPs: 133,577,729,638,400
transformer_blocks(98.03%): 130,942,815,436,800
|- Multihead_Self_Attention(73.79%): 98,569,499,443,200
  |- q_proj(3.01%): 4,026,531,840,000
  |- k_proj(3.01%): 4,026,531,840,000
  |- v_proj(3.01%): 4,026,531,840,000
  |- output_proj(3.01%): 4,026,531,840,000
  |- QK^T(30.87%): 41,231,686,041,600
  |- softmax(QK^T)V(30.87%): 41,231,686,041,600
|- FFN(24.24%): 32,373,315,993,600
  |- w1(8.08%): 10,791,105,331,200
  |- w2(8.08%): 10,791,105,331,200
  |- w3(8.08%): 10,791,105,331,200
lm_head(1.97%): 2,634,914,201,600
```

相比较之前的
```(context_length: 1,024)
total FLOPs: 3,516,769,894,400
transformer_blocks(95.32%): 3,352,087,756,800
|- Multihead_Self_Attention(37.78%): 1,328,755,507,200
  |- q_proj(7.16%): 251,658,240,000
  |- k_proj(7.16%): 251,658,240,000
  |- v_proj(7.16%): 251,658,240,000
  |- output_proj(7.16%): 251,658,240,000
  |- QK^T(4.58%): 161,061,273,600
  |- softmax(QK^T)V(4.58%): 161,061,273,600
|- FFN(57.53%): 2,023,332,249,600
  |- w1(19.18%): 674,444,083,200
  |- w2(19.18%): 674,444,083,200
  |- w3(19.18%): 674,444,083,200
lm_head(4.68%): 164,682,137,600
```
自注意力层明显占比上升，尤其是计算QK^T和softmax(QK^T)V的部分