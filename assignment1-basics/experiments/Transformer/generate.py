from cs336_basics.generation import decode

model_config_path = "checkpoints/tinystories-bs128-lr-2e-3-327680000/config.json"
checkpoint_path = "checkpoints/tinystories-bs128-lr-2e-3-327680000/latest.pt"
prompt = "Once upon a time"
max_new_tokens = 256
temperature = 0.1
top_p = 0.6
device = "cuda:6"
vocab_filepath = "results/TinyStoriesV2-GPT4-vocab.txt"
merges_filepath = "results/TinyStoriesV2-GPT4-merges.txt"
special_tokens = ["<|endoftext|>"]
print(
    decode(
        model_config_path,
        checkpoint_path,
        prompt,
        max_new_tokens,
        temperature,
        top_p,
        device,
        vocab_filepath,
        merges_filepath,
        special_tokens,
    )
)
