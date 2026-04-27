## Problem (adamw_accounting): Resource accounting for training with AdamW (2 points)

### Prompt

Let us compute how much memory and compute running AdamW requires. Assume we are using `float32` for every tensor.

(a) How much peak memory does running AdamW require? Decompose your answer based on the memory usage of the parameters, activations, gradients, and optimizer state. Express your answer in terms of `batch_size` and the model hyperparameters (`vocab_size`, `context_length`, `num_layers`, `d_model`, `num_heads`). Assume `d_ff = 8 / 3 × d_model`.

For simplicity, when calculating memory usage of activations, consider only the following components:

- Transformer block: RMSNorm(s)
- Transformer block: multi-head self-attention sublayer (`QKV` projections, `QK^T` matrix multiply, softmax, weighted sum of values, output projection)
- Transformer block: position-wise feed-forward (SwiGLU): `W1`, `W2`, SiLU on the gate branch, element-wise product, `W3`
- Final RMSNorm
- Output embedding
- Cross-entropy on logits

> Deliverable: An algebraic expression for each of parameters, activations, gradients, and optimizer state, as well as the total.

(b) Instantiate your answer for a GPT-2 XL-shaped model to get an expression that only depends on `batch_size`. What is the maximum batch size you can use and still fit within `80 GB` memory?
For simplicity, when calculating memory usage of activations, consider only the following 
components:
• Transformer block
‣RMSNorm(s)
‣Multi-head self-attention sublayer: 𝑄𝐾𝑉  projections, 𝑄𝐾⊤ matrix multiply, softmax, 
weighted sum of values, output projection.
‣Position-wise feed-forward (SwiGLU): 𝑊1, 𝑊2, SiLU on the gate branch, element-wise 
product, 𝑊3
32
• final RMSNorm
• output embedding
• cross-entropy on logits
Deliverable: An algebraic expression for each of parameters, activations, gradients, and 
optimizer state, as well as the total
> Deliverable: An expression that looks like `a ⋅ batch_size + b` for numerical values `a`, `b`, and a number representing the maximum batch size.

(c) How many FLOPs does running one step of AdamW take?

> Deliverable: An algebraic expression, with a brief justification.

(d) Model FLOPs utilization (MFU) is defined as the ratio of observed throughput, tokens per second, relative to the hardware’s theoretical peak FLOP throughput [A. Chowdhery et al., 2022]. An NVIDIA H100 GPU has a theoretical peak of `495 teraFLOP/s` for “float32”, actually TensorFloat-32, which in reality is “bfloat19”, operations. Assuming you are able to get `50% MFU`, how long would it take to train a GPT-2 XL for `400K` steps and a batch size of `1024` on a single H100? Following J. Kaplan et al. [25] and J. Hoffmann et al. [26], assume that the backward pass has twice the FLOPs of the forward pass.

> Deliverable: The number of hours training would take, with a brief justification.

### Answer

a. 
- input embedding
    - parameters
        - `vocab_size * d_model`
    - activations(not consider in this problem)
    - gradients
        - `vocab_size * d_model`
    - optimizer state
        - `2 * (vocab_size * d_model)`
- Transformer block
    - 2 * RMSNorm 
        - parameters
            - `2 * (d_model)`
        - activations
            - rms
                - `2 * (batch_size * context_length * 1)`
            - output
                - `2 * (batch_size * context_length * d_model)`
        - gradients
            - `2 * (d_model)`
        - optimizer state
            - `2 * (2 * d_model)`
    - Multi-head self-attention sublayer
        - `QKV` projections
            - parameters
                - `3 * (d_model * d_model)`
            - activations
                - `3 * (batch_size * context_length * d_model)`
            - gradients
                - `3 * (d_model * d_model)`
            - optimizer state
                - `6 * (d_model * d_model)`
        - `QK^T` matrix multiply (Q: [batch_size num_heads context_length hd_k], K: [batch_size num_heads context_length hd_k])
            - activations
                - `batch_size * num_heads * context_length * context_length`
        - softmax
            - activations
                - `batch_size * num_heads * context_length * context_length`
        - weighted sum of values([batch_size num_heads context_length context_length], [batch_size num_heads context_length hd_v])
            - activations
                - `batch_size * context_length * d_model`
        - output projection
            - parameters
                - `d_model * d_model`
            - activations
                - `batch_size * context_length * d_model`
            - gradients
                - `d_model * d_model`
            - optimizer state
                - `2 * d_model * d_model`
    - Position-wise feed-forward (SwiGLU)
        - gate projection
            - parameters
                - `d_model * d_ff`
            - activations
                - `batch_size * context_length * d_ff`
            - gradients
                - `d_model * d_ff`
            - optimizer state
                - `2 * d_model * d_ff`
        - value projection
            - parameters
                - `d_model * d_ff`
            - activations
                - `batch_size * context_length * d_ff`
            - gradients
                - `d_model * d_ff`
            - optimizer state
                - `2 * d_model * d_ff`
        - SiLU on the gate branch
            - activations
                - `batch_size * context_length * d_ff`
        - element-wise product
            - activations
                - `batch_size * context_length * d_ff`
        - down projection
            - parameters
                - `d_model * d_ff`
            - activations
                - `batch_size * context_length * d_model`
            - gradients
                - `d_model * d_ff`
            - optimizer state
                - `2 * d_model * d_ff`
- final RMSNorm
    - parameters
        - `d_model`
    - activations
        - rms
            - `batch_size * context_length * 1`
        - output
            - `batch_size * context_length * d_model`
    - gradients
        - `d_model`
    - optimizer state
        - `2 * d_model`
- Output embedding
    - parameters
        - `d_model * vocab_size`
    - activations
        - `batch_size * context_length * vocab_size`
    - gradients
        - `d_model * vocab_size`
    - optimizer state
        - `2 * d_model * vocab_size`
- Cross-entropy on logits
    - activations
        - max-logits
            - `batch_size * context_length`
        - exp_shifted
            - `batch_size * context_length * vocab_size`
        - exp_shifted_sum
            - `batch_size * context_length`
        - correct_logits
            - `batch_size * context_length`

parameters memory: `4*D*(2*L*(6*D + 1) + 2*V + 1)`
activations memory: `4*B*T*(3*D + 2*L*(28*D + 3*H*T + 3) + 6*V + 12)/3`
gradients memory: `4*D*(2*L*(6*D + 1) + 2*V + 1)`
optimizer_state memory: `8*D*(2*L*(6*D + 1) + 2*V + 1)`
total memory: `4*B*T*(3*D + 2*L*(28*D + 3*H*T + 3) + 6*V + 12)/3 + 16*D*(2*L*(6*D + 1) + 2*V + 1)`

b.  GPT-2 XL-shaped model memory: 16357023744*B + 26168601600 bytes
    maximum batch size within 80GB memory = 3
c. AdamW optimizer update FLOPs: 14 * num_parameters
d. about 4850 hours