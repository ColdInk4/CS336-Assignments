import sympy as sp

# 定义符号变量
batch_size, vocab_size, context_length, num_layers, d_model, num_heads = sp.symbols(
    "B V T L D H"
)
d_ff = sp.Rational(8, 3) * d_model
parameters_expr = (
    vocab_size * d_model
    + num_layers
    * (
        2 * (d_model)
        + 3 * (d_model * d_model)
        + d_model * d_model
        + d_model * d_ff
        + d_model * d_ff
        + d_model * d_ff
    )
    + d_model
    + d_model * vocab_size
)
activations_expr = (
    num_layers
    * (
        2 * (batch_size * context_length)
        + 2 * (batch_size * context_length * d_model)
        + 3 * (batch_size * context_length * d_model)
        + batch_size * num_heads * context_length * context_length
        + batch_size * num_heads * context_length * context_length
        + batch_size * context_length * d_model
        + batch_size * context_length * d_model
        + batch_size * context_length * d_ff
        + batch_size * context_length * d_ff
        + batch_size * context_length * d_ff
        + batch_size * context_length * d_ff
        + batch_size * context_length * d_model
    )
    + batch_size * context_length
    + batch_size * context_length * d_model
    + batch_size * context_length * vocab_size
    + batch_size * context_length
    + batch_size * context_length * vocab_size
    + batch_size * context_length
    + batch_size * context_length
)

parameters_memory = sp.simplify(parameters_expr * 4)
activations_memory = sp.simplify(activations_expr * 4)
gradients_memory = parameters_memory
optimizer_state_memory = 2 * parameters_memory
total_memory = sp.simplify(
    parameters_memory + activations_memory + gradients_memory + optimizer_state_memory
)
print(f"parameters memory: {parameters_memory}")
print(f"activations memory: {activations_memory}")
print(f"gradients memory: {gradients_memory}")
print(f"optimizer_state memory: {optimizer_state_memory}")
print(f"total memory: {total_memory}")

# V:  50,257
# T:  1,024
# L:  48
# D:  1,600
# H:  25
GPT2_XLshaped_model_memory = total_memory.subs(
    {
        vocab_size: 50257,
        context_length: 1024,
        num_layers: 48,
        d_model: 1600,
        num_heads: 25,
    }
)
print(f"GPT-2 XL-shaped model memory: {GPT2_XLshaped_model_memory}")
print(
    f"maximum batch size within 80GB memory: {sp.solve(GPT2_XLshaped_model_memory<80*1000*1000*1000)}"
)
update_FLOPs_expr = sp.simplify(14 * parameters_expr)
update_FLOPs = update_FLOPs_expr.subs(
    {
        vocab_size: 50257,
        context_length: 1024,
        num_layers: 48,
        d_model: 1600,
        num_heads: 25,
        batch_size: 1024,
    }
)
forward_FLOPs = (
    1024 * 3516769894400
)  # accounting in writeup/P6_transformer_accounting.md
backford_FLOPs = 2 * forward_FLOPs
total_FLOPs = (forward_FLOPs + backford_FLOPs + update_FLOPs) * 400000
MPU_FLOP_per_sec = 495 * 1000 * 1000 * 1000 * 1000 / 2
print(f"Time is about {total_FLOPs/MPU_FLOP_per_sec/60/60} hours")
