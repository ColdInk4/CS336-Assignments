from cs336_basics.bpe import train_bpe_from_filepath

vocab, merges = train_bpe_from_filepath(
    "data/TinyStoriesV2-GPT4-train.txt",
    10000,
    ["<|endoftext|>"],
    64,
)

with open("results/TinyStoriesV2-GPT4-vocab.txt", "w") as f:
    for token_id, token_bytes in vocab.items():
        f.write(f"{token_id}\t{token_bytes.hex()}\n")

with open("results/readable/TinyStoriesV2-GPT4-vocab-readable.txt", "w") as f:
    for token_id, token_bytes in vocab.items():
        f.write(f"{token_id}: {token_bytes}\n")

with open("results/TinyStoriesV2-GPT4-merges.txt", "w") as f:
    for left_bytes, right_bytes in merges:
        f.write(f"{left_bytes.hex()}\t{right_bytes.hex()}\n")

with open("results/readable/TinyStoriesV2-GPT4-merges-readable.txt", "w") as f:
    for left_bytes, right_bytes in merges:
        f.write(f"({left_bytes}, {right_bytes})\n")

longest_token_bytes = max(vocab.values(), key=len)
length = len(longest_token_bytes)

print(f"Longest token (bytes representation): {longest_token_bytes}")
print(f"Length in bytes: {length}")

try:
    decoded_str = longest_token_bytes.decode("utf-8")
    print(f"Decoded string: '{decoded_str}'")
except UnicodeDecodeError:
    print("Cannot decode as UTF-8 (this is normal for some intermediate BPE bytes)")
