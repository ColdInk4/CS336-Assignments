from cs336_basics.BPE.pretokenization import pretokenize
import os


def init_vocab(vocab_size: int, special_tokens: list[str]) -> dict[int, bytes]:
    if vocab_size < 256 + len(special_tokens):
        raise ValueError(
            "vocab_size 小于“初始 256 bytes 加 special tokens 所需的最小词表大小"
        )
    return {i: bytes([i]) for i in range(256)} | {
        (256 + i): special_tokens[i].encode("utf-8") for i in range(len(special_tokens))
    }


def train_bpe(
    vocab_size: int,
    vocab_table: dict[int, bytes],
    frequency_table: dict[tuple[bytes, ...], int],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    merge_round: int = vocab_size - len(vocab_table)
    merges: list[tuple[bytes, bytes]] = []

    for _ in range(merge_round):
        bind_flag = False  # 处理“已经没有可 merge pair” 的情况
        pair_frequency_table: dict[tuple[bytes, bytes], int] = {}

        for word, frequency in frequency_table.items():
            for pair_index in range(len(word) - 1):
                bind_flag = True
                bind = (word[pair_index], word[pair_index + 1])
                pair_frequency_table[bind] = (
                    pair_frequency_table.get(bind, 0) + frequency
                )
        if bind_flag == False:
            break

        max_pair, _ = max(pair_frequency_table.items(), key=lambda x: (x[1], x[0]))
        merges.append(max_pair)
        new_vocab = max_pair[0] + max_pair[1]
        vocab_table[len(vocab_table)] = new_vocab

        new_frequency_table = {}
        for word, frequency in frequency_table.items():
            new_word = ()
            pair_index = 0
            while pair_index < len(word) - 1:
                bind = (word[pair_index], word[pair_index + 1])
                if bind == max_pair:
                    new_word += (new_vocab,)
                    pair_index += 2
                else:
                    new_word += (word[pair_index],)
                    pair_index += 1
            if pair_index == len(word) - 1:
                new_word += (word[pair_index],)
            new_frequency_table[new_word] = (
                new_frequency_table.get(new_word, 0) + frequency
            )

        frequency_table = new_frequency_table
    return (vocab_table, merges)


def train_bpe_from_filepath(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab_table: dict[int, bytes] = init_vocab(vocab_size, special_tokens)
    frequency_table: dict[tuple[bytes, ...], int] = pretokenize(
        input_path, special_tokens, num_processes
    )

    return train_bpe(vocab_size, vocab_table, frequency_table)
