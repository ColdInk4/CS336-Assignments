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
    frequency_table: dict[tuple[bytes, ...], int],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    merge_round: int = vocab_size - len(vocab_table)
    merges: list[tuple[bytes, bytes]] = []
    pair_count: dict[tuple[bytes, bytes], int] = defaultdict(
        int
    )  # 每个 pair 的全局频次
    pair_from_pre_token: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = (
        defaultdict(set)
    )  # 每个 pair 在哪些 pre_token 内部出现

    for pre_token, frequency in frequency_table.items():
        for pair_index in range(len(pre_token) - 1):
            pair: tuple[bytes, bytes] = (
                pre_token[pair_index],
                pre_token[pair_index + 1],
            )
            pair_count[pair] += frequency
            pair_from_pre_token[pair].add(pre_token)

    for _ in range(merge_round):
        # 目前最大频次对应的 pair
        # print("frequency_table: ", frequency_table)
        # print(pair_count)
        # print(pair_from_pre_token)
        if len(pair_count) == 0:
            break
        max_pair, _ = max(pair_count.items(), key=lambda x: (x[1], x[0]))
        merges.append(max_pair)

        # 合并后的新词
        new_vocab: bytes = max_pair[0] + max_pair[1]
        vocab_table[len(vocab_table)] = new_vocab

        # 目前最大频次对应的 pair 转换为list
        max_pair_list = list(max_pair)
        max_pair_from_pre_token = pair_from_pre_token[max_pair]
        for pre_token in max_pair_from_pre_token:
            new_pre_token_list = []
            pair_index = 0

            # 发生改变的index
            index_changed_old = set()  # 从这个index开始的pair，要进行减
            index_changed_new = set()  # 从这个index开始的pair，要进行加

            while pair_index < len(pre_token) - 1:
                pair_list = [pre_token[pair_index], pre_token[pair_index + 1]]
                # print(pair_list)
                if pair_list == max_pair_list:
                    new_pre_token_list.append(new_vocab)
                    new_pair_index = len(new_pre_token_list) - 1
                    # 受影响的项是index左右的
                    # loweweabcwest   index_changed_old
                    #  +^^^^+ +^^+
                    # loAAabcAst   index_changed_new
                    #  +^^+ +^+
                    if pair_index != 0:
                        index_changed_old.add(
                            pair_index - 1,
                        )
                        index_changed_new.add(new_pair_index - 1)
                    index_changed_old.add(
                        pair_index,
                    )
                    if pair_index != len(pre_token) - 2:
                        index_changed_old.add(
                            pair_index + 1,
                        )
                        index_changed_new.add(new_pair_index)

                    pair_index += 2
                else:
                    new_pre_token_list.append(pre_token[pair_index])
                    pair_index += 1
            if pair_index == len(pre_token) - 1:
                new_pre_token_list.append(pre_token[pair_index])

            new_pre_token = tuple(new_pre_token_list)  # 新的pre_token

            # 进行 pair_from_pre_token 的更新
            for old_index in range(len(pre_token) - 1):
                pair = (pre_token[old_index], pre_token[old_index + 1])
                if pair != max_pair and pre_token in pair_from_pre_token[pair]:
                    pair_from_pre_token[pair].remove(pre_token)

                if old_index not in index_changed_old:
                    pair_from_pre_token[pair].add(new_pre_token)
                else:
                    pair_count[pair] -= frequency_table[pre_token]
                    if pair_count[pair] == 0:
                        del pair_count[pair]
                        del pair_from_pre_token[pair]

            for new_index in index_changed_new:
                pair = (new_pre_token[new_index], new_pre_token[new_index + 1])
                pair_count[pair] += frequency_table[pre_token]
                pair_from_pre_token[pair].add(new_pre_token)

            frequency_table[new_pre_token] += frequency_table[pre_token]
            del frequency_table[pre_token]

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
