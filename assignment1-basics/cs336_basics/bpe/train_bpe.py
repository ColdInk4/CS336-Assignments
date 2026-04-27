from cs336_basics.bpe.pretokenization import pretokenize
import os
from collections import defaultdict
import heapq
import cProfile
import pstats

BPE_PROFILE = 0


class Entry:
    def __init__(self, count: int, pair: tuple[bytes, bytes]):
        self.count = count
        self.pair = pair

    def __repr__(self):
        return f"({self.count}, {self.pair})"

    def __iter__(self):
        yield self.count
        yield self.pair

    def __lt__(self, other):
        return (
            self.count > other.count
            or self.count == other.count
            and self.pair > other.pair
        )


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

    if BPE_PROFILE:
        print("====Start Training BPE====")
        profiler = cProfile.Profile()
        profiler.enable()

    num_merges: int = vocab_size - len(vocab_table)
    merges: list[tuple[bytes, bytes]] = []
    global_pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(
        int
    )  # 每个 pair 的全局频次
    pair_to_pretokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(
        set
    )  # 每个 pair 在哪些 pretoken 内部出现

    pair_counts_heap = []

    for pretoken, frequency in current_pretoken_freqs.items():
        for old_idx in range(len(pretoken) - 1):
            pair: tuple[bytes, bytes] = (
                pretoken[old_idx],
                pretoken[old_idx + 1],
            )
            global_pair_counts[pair] += frequency
            pair_to_pretokens[pair].add(pretoken)

    for pair, frequency in global_pair_counts.items():
        heapq.heappush(pair_counts_heap, Entry(frequency, pair))

    pair_to_pretokens_removals = defaultdict(set)
    pair_to_pretokens_additions = defaultdict(set)
    pair_count_deltas = defaultdict(int)
    pretoken_freq_deltas = defaultdict(int)
    pretokens_to_remove_from_freqs = set()

    for _ in range(num_merges):
        if len(global_pair_counts) == 0:
            break
        max_count, max_pair = heapq.heappop(pair_counts_heap)

        while (
            max_pair not in global_pair_counts
            or global_pair_counts[max_pair] != max_count
        ):
            max_count, max_pair = heapq.heappop(pair_counts_heap)
        merges.append(max_pair)

        # 合并后的新词
        merged_token_bytes: bytes = max_pair[0] + max_pair[1]
        vocab_table[len(vocab_table)] = merged_token_bytes

        pair_to_pretokens_removals.clear()
        pair_to_pretokens_additions.clear()
        pair_count_deltas.clear()
        pretoken_freq_deltas.clear()
        pretokens_to_remove_from_freqs.clear()

        for pretoken in pair_to_pretokens[max_pair]:
            new_pretoken_parts = []
            old_idx = 0

            # 发生改变的index
            changed_old_pair_starts = set()  # 从这个index开始的pair，要进行减
            changed_new_pair_starts = set()  # 从这个index开始的pair，要进行加

            while old_idx < len(pretoken) - 1:
                candidate_pair = (pretoken[old_idx], pretoken[old_idx + 1])
                if candidate_pair == max_pair:
                    new_pretoken_parts.append(merged_token_bytes)
                    new_pair_index = len(new_pretoken_parts) - 1
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
                    new_pretoken_parts.append(pretoken[old_idx])
                    old_idx += 1
            if old_idx == len(pretoken) - 1:
                new_pretoken_parts.append(pretoken[old_idx])

            merged_pretoken = tuple(new_pretoken_parts)  # 新的pretoken

            # 进行 pair_from_pretoken 的更新
            for old_index in range(len(pretoken) - 1):
                pair = (pretoken[old_index], pretoken[old_index + 1])
                if pretoken in pair_to_pretokens[pair]:
                    pair_to_pretokens_removals[pair].add(pretoken)

                if old_index not in changed_old_pair_starts:
                    pair_to_pretokens_additions[pair].add(merged_pretoken)
                else:
                    pair_count_deltas[pair] -= current_pretoken_freqs[pretoken]

            for new_index in changed_new_pair_starts:
                pair = (merged_pretoken[new_index], merged_pretoken[new_index + 1])
                pair_count_deltas[pair] += current_pretoken_freqs[pretoken]
                pair_to_pretokens_additions[pair].add(merged_pretoken)

            pretoken_freq_deltas[merged_pretoken] += current_pretoken_freqs[pretoken]
            pretokens_to_remove_from_freqs.add(pretoken)

        for pair, pretokens in pair_to_pretokens_removals.items():
            for pretoken in pretokens:
                pair_to_pretokens[pair].remove(pretoken)

        for pair, pretokens in pair_to_pretokens_additions.items():
            for pretoken in pretokens:
                pair_to_pretokens[pair].add(pretoken)

        for pair, count in pair_count_deltas.items():
            global_pair_counts[pair] += count
            if global_pair_counts[pair] == 0:
                del global_pair_counts[pair]
                del pair_to_pretokens[pair]
            else:
                heapq.heappush(pair_counts_heap, Entry(global_pair_counts[pair], pair))

        for pretoken, freq in pretoken_freq_deltas.items():
            current_pretoken_freqs[pretoken] += freq

        for pretoken in pretokens_to_remove_from_freqs:
            del current_pretoken_freqs[pretoken]

    if BPE_PROFILE:
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumtime").print_stats(20)

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


if __name__ == "__main__":
    train_bpe_from_filepath(
        "data/TinyStoriesV2-GPT4-train.txt",
        10000,
        ["<|endoftext|>"],
        64,
    )
