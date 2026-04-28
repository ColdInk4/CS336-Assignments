import json
from cs336_basics.transformer import Transformer_LM, softmax
from cs336_basics.training import load_model_checkpoint
from cs336_basics.bpe import Tokenizer
import torch
from jaxtyping import Int, Float
from torch import Tensor
from einops import rearrange


def decode(
    model_config_path,
    checkpoint_path,
    prompt,
    max_new_tokens,
    temperature,
    top_p,
    device,
    vocab_filepath,
    merges_filepath,
    special_tokens: list[str] | None = None,
):
    if top_p <= 0 or top_p > 1:
        raise ValueError("p must fit 0 < p <= 1")
    if temperature < 0:
        raise ValueError("temperature must >= 0")
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)

    with open(model_config_path, "r") as f:
        config = json.load(f)

    model = Transformer_LM(
        config["model"]["d_model"],
        config["model"]["num_heads"],
        config["model"]["d_ff"],
        config["model"]["vocab_size"],
        config["model"]["context_length"],
        config["model"]["num_layers"],
        config["model"]["theta"],
        device,
    )

    load_model_checkpoint(checkpoint_path, model, map_location=device)

    prompt_ids: Int[Tensor, "prompt_length"] = torch.tensor(
        tokenizer.encode(prompt), device=device, dtype=torch.long
    )
    generated_ids: Int[Tensor, "1 prompt_length"] = rearrange(
        prompt_ids, "prompt_length->1 prompt_length"
    )

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context_ids = generated_ids[:, -config["model"]["context_length"] :]
            logits: Float[Tensor, "1 current_context_length vocab_size"] = model(
                context_ids
            )
            next_token_logits: Float[Tensor, "vocab_size"] = logits[0, -1]
            if temperature > 0:
                next_token_probs: Float[Tensor, "vocab_size"] = softmax(
                    next_token_logits / temperature, dim=-1
                )
                sorted_probs, sorted_indices = next_token_probs.sort(
                    dim=-1, descending=True
                )
                cumsum_probs = sorted_probs.cumsum(dim=-1)
                above = cumsum_probs >= top_p
                cutoff = torch.nonzero(above)[0].item()
                nucleus_probs = sorted_probs[: cutoff + 1]
                nucleus_indices = sorted_indices[: cutoff + 1]
                nucleus_probs = nucleus_probs / nucleus_probs.sum()

                sampled_position = int(torch.multinomial(nucleus_probs, 1).item())
                next_token_id = nucleus_indices[sampled_position]
            else:
                next_token_id = next_token_logits.argmax()
            next_token_tensor = rearrange(next_token_id, "->1 1")
            generated_ids = torch.concat(
                [generated_ids, next_token_tensor],
                dim=-1,
            )

            if int(next_token_id.item()) in tokenizer.special_token_ids:
                break
    full_text = tokenizer.decode(generated_ids[0].tolist())
    return full_text
