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
    )  # 每个 pair 在哪些 pre_token 内部出现

    for pre_token, frequency in current_pretoken_freqs.items():
        for old_idx in range(len(pre_token) - 1):
            pair: tuple[bytes, bytes] = (
                pre_token[old_idx],
                pre_token[old_idx + 1],
            )
            global_pair_counts[pair] += frequency
            pair_to_pretokens[pair].add(pre_token)

    for _ in range(num_merges):
        # 目前最大频次对应的 pair
        # print("frequency_table: ", frequency_table)
        # print(pair_count)
        # print(pair_from_pre_token)
        if len(global_pair_counts) == 0:
            break
        max_pair, _ = max(global_pair_counts.items(), key=lambda x: (x[1], x[0]))
        merges.append(max_pair)

        # 合并后的新词
        merged_token_bytes: bytes = max_pair[0] + max_pair[1]
        vocab_table[len(vocab_table)] = merged_token_bytes

        # 目前最大频次对应的 pair 转换为list
        tokens_with_max_pair = pair_to_pretokens[max_pair]
        for pre_token in tokens_with_max_pair:
            merged_pretoken_parts = []
            old_idx = 0

            # 发生改变的index
            changed_old_pair_starts = set()  # 从这个index开始的pair，要进行减
            changed_new_pair_starts = set()  # 从这个index开始的pair，要进行加

            while old_idx < len(pre_token) - 1:
                pair_list = (pre_token[old_idx], pre_token[old_idx + 1])
                # print(pair_list)
                if pair_list == max_pair:
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
                    if old_idx != len(pre_token) - 2:
                        changed_old_pair_starts.add(
                            old_idx + 1,
                        )
                        changed_new_pair_starts.add(new_pair_index)

                    old_idx += 2
                else:
                    merged_pretoken_parts.append(pre_token[old_idx])
                    old_idx += 1
            if old_idx == len(pre_token) - 1:
                merged_pretoken_parts.append(pre_token[old_idx])

            merged_pretoken = tuple(merged_pretoken_parts)  # 新的pre_token

            # 进行 pair_from_pre_token 的更新
            for old_index in range(len(pre_token) - 1):
                pair = (pre_token[old_index], pre_token[old_index + 1])
                if pair != max_pair and pre_token in pair_to_pretokens[pair]:
                    pair_to_pretokens[pair].remove(pre_token)

                if old_index not in changed_old_pair_starts:
                    pair_to_pretokens[pair].add(merged_pretoken)
                else:
                    global_pair_counts[pair] -= current_pretoken_freqs[pre_token]
                    if global_pair_counts[pair] == 0:
                        del global_pair_counts[pair]
                        del pair_to_pretokens[pair]

            for new_index in changed_new_pair_starts:
                pair = (merged_pretoken[new_index], merged_pretoken[new_index + 1])
                global_pair_counts[pair] += current_pretoken_freqs[pre_token]
                pair_to_pretokens[pair].add(merged_pretoken)

            current_pretoken_freqs[merged_pretoken] += current_pretoken_freqs[pre_token]
            del current_pretoken_freqs[pre_token]

        # print(pair_count)
        # print(pair_from_pre_token)

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
