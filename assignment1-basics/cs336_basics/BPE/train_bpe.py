from cs336_basics.BPE.pretokenization import pretokenize
import os
from collections import defaultdict


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
    current_pretoken_freqs: dict[tuple[bytes, ...], int],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    num_merges: int = vocab_size - len(vocab_table)
    merges: list[tuple[bytes, bytes]] = []
    global_pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(
        int
    )  # 每个 pair 的全局频次
    pair_to_pretokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(
        set
    )  # 每个 pair 在哪些 pretoken 内部出现

    for pretoken, frequency in current_pretoken_freqs.items():
        for old_idx in range(len(pretoken) - 1):
            pair: tuple[bytes, bytes] = (
                pretoken[old_idx],
                pretoken[old_idx + 1],
            )
            global_pair_counts[pair] += frequency
            pair_to_pretokens[pair].add(pretoken)

    for _ in range(num_merges):
        if len(global_pair_counts) == 0:
            break
        max_pair, _ = max(global_pair_counts.items(), key=lambda x: (x[1], x[0]))
        merges.append(max_pair)

        # 合并后的新词
        merged_token_bytes: bytes = max_pair[0] + max_pair[1]
        vocab_table[len(vocab_table)] = merged_token_bytes

        affected_pretokens = pair_to_pretokens[max_pair]

        pretokens_to_be_removed = defaultdict(set)
        pretokens_to_be_added = defaultdict(set)
        global_pair_counts_to_be_changed = defaultdict(int)
        current_pretoken_freqs_to_be_changed = defaultdict(int)
        current_pretoken_freqs_to_be_removed = set()

        for pretoken in affected_pretokens:
            merged_pretoken_parts = []
            old_idx = 0

            # 发生改变的index
            changed_old_pair_starts = set()  # 从这个index开始的pair，要进行减
            changed_new_pair_starts = set()  # 从这个index开始的pair，要进行加

            while old_idx < len(pretoken) - 1:
                candidate_pair = (pretoken[old_idx], pretoken[old_idx + 1])
                if candidate_pair == max_pair:
                    merged_pretoken_parts.append(merged_token_bytes)
                    new_pair_index = len(merged_pretoken_parts) - 1
                    # 受影响的项是index左右的
                    # loweweabcwest   index_changed_old
                    #  +^^^^+ +^^+
                    # loAAabcAst   index_changed_new
                    #  +^^+ +^+
                    if old_idx != 0:
                        changed_old_pair_starts.add(
                            old_idx - 1,
                        )
                        changed_new_pair_starts.add(new_pair_index - 1)
                    changed_old_pair_starts.add(
                        old_idx,
                    )
                    if old_idx != len(pretoken) - 2:
                        changed_old_pair_starts.add(
                            old_idx + 1,
                        )
                        changed_new_pair_starts.add(new_pair_index)

                    old_idx += 2
                else:
                    merged_pretoken_parts.append(pretoken[old_idx])
                    old_idx += 1
            if old_idx == len(pretoken) - 1:
                merged_pretoken_parts.append(pretoken[old_idx])

            merged_pretoken = tuple(merged_pretoken_parts)  # 新的pretoken

            # 进行 pair_from_pretoken 的更新
            for old_index in range(len(pretoken) - 1):
                pair = (pretoken[old_index], pretoken[old_index + 1])
                if pretoken in pair_to_pretokens[pair]:
                    pretokens_to_be_removed[pair].add(pretoken)

                if old_index not in changed_old_pair_starts:
                    pretokens_to_be_added[pair].add(merged_pretoken)
                else:
                    global_pair_counts_to_be_changed[pair] -= current_pretoken_freqs[
                        pretoken
                    ]

            for new_index in changed_new_pair_starts:
                pair = (merged_pretoken[new_index], merged_pretoken[new_index + 1])
                global_pair_counts_to_be_changed[pair] += current_pretoken_freqs[
                    pretoken
                ]
                pretokens_to_be_added[pair].add(merged_pretoken)

            current_pretoken_freqs_to_be_changed[
                merged_pretoken
            ] += current_pretoken_freqs[pretoken]
            current_pretoken_freqs_to_be_removed.add(pretoken)

        for pair, pretokens in pretokens_to_be_removed.items():
            for pretoken in pretokens:
                pair_to_pretokens[pair].remove(pretoken)

        for pair, pretokens in pretokens_to_be_added.items():
            for pretoken in pretokens:
                pair_to_pretokens[pair].add(pretoken)

        for pair, count in global_pair_counts_to_be_changed.items():
            global_pair_counts[pair] += count
            if global_pair_counts[pair] == 0:
                del global_pair_counts[pair]
                del pair_to_pretokens[pair]

        for pretoken, freq in current_pretoken_freqs_to_be_changed.items():
            current_pretoken_freqs[pretoken] += freq

        for pretoken in current_pretoken_freqs_to_be_removed:
            del current_pretoken_freqs[pretoken]

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
