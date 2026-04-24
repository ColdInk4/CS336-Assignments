vocab_size, context_length = [50257, 16384]

GPT_2_small_sized = [12, 768, 12]
GPT_2_medium_sized = [24, 1024, 16]
GPT_2_large_sized = [36, 1280, 20]
GPT_2_XL_sized = [48, 1600, 25]

num_layers, d_model, num_heads = GPT_2_XL_sized

d_ff = round(8 / 3 * d_model / 64) * 64
print(f"d_ff: {d_ff:,}")

trainable_parameters = (
    2 * vocab_size * d_model
    + num_layers * (4 * d_model * d_model + 2 * d_model + 3 * d_model * d_ff)
    + d_model
)
print(f"trainable_parameters: {trainable_parameters:,}")

FLOPs = (
    num_layers
    * (
        8 * d_model * d_model * context_length
        + 4 * context_length * context_length * d_model
        + 6 * d_model * d_ff * context_length
    )
    + 2 * d_model * vocab_size * context_length
)
print(f"total FLOPs: {FLOPs:,}")
transformer_blocks_flop = num_layers * (
    8 * d_model * d_model * context_length
    + 4 * (context_length * context_length) * d_model
    + 6 * d_model * d_ff * context_length
)
Multihead_Self_Attention_flop = num_layers * (
    8 * d_model * d_model * context_length
    + 4 * (context_length * context_length) * d_model
)
q_proj = num_layers * (2 * d_model * d_model * context_length)
k_proj = num_layers * (2 * d_model * d_model * context_length)
v_proj = num_layers * (2 * d_model * d_model * context_length)
output_proj = num_layers * (2 * d_model * d_model * context_length)
Q_K = num_layers * (2 * (context_length * context_length) * d_model)
Q_K_V = num_layers * (2 * (context_length * context_length) * d_model)

FFN_flop = num_layers * (6 * d_model * d_ff * context_length)
w1 = num_layers * (2 * d_model * d_ff * context_length)
w2 = num_layers * (2 * d_model * d_ff * context_length)
w3 = num_layers * (2 * d_model * d_ff * context_length)

lm_head_flop = 2 * d_model * vocab_size * context_length
print(
    f"transformer_blocks({transformer_blocks_flop / FLOPs:.2%}): {transformer_blocks_flop:,}"
)
print(
    f"|- Multihead_Self_Attention({Multihead_Self_Attention_flop / FLOPs:.2%}): {Multihead_Self_Attention_flop:,}"
)
print(f"  |- q_proj({q_proj / FLOPs:.2%}): {q_proj:,}")
print(f"  |- k_proj({k_proj / FLOPs:.2%}): {k_proj:,}")
print(f"  |- v_proj({v_proj / FLOPs:.2%}): {v_proj:,}")
print(f"  |- output_proj({output_proj / FLOPs:.2%}): {output_proj:,}")
print(f"  |- QK^T({Q_K / FLOPs:.2%}): {Q_K:,}")
print(f"  |- softmax(QK^T)V({Q_K_V  / FLOPs:.2%}): {Q_K_V :,}")
print(f"|- FFN({FFN_flop / FLOPs:.2%}): {FFN_flop:,}")
print(f"  |- w1({w1 / FLOPs:.2%}): {w1:,}")
print(f"  |- w2({w2 / FLOPs:.2%}): {w2:,}")
print(f"  |- w3({w3 / FLOPs:.2%}): {w3:,}")
print(f"lm_head({lm_head_flop / FLOPs:.2%}): {lm_head_flop:,}")
