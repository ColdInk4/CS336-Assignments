import regex as re
from collections.abc import Iterable, Iterator
from tests.common import gpt2_bytes_to_unicode


class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):

        self.vocab: dict[int, bytes] = vocab
        self.merges: list[tuple[bytes, bytes]] = merges
        self.special_tokens: list[str] | None = special_tokens
        self.special_tokens_len_max = (
            len(max(self.special_tokens, key=len)) if self.special_tokens else 0
        )
        self.bytes_to_id: dict[bytes, int] = {v: k for k, v in vocab.items()}
        self.merge_rank: dict[tuple[bytes, bytes], int] = {
            merge: index for index, merge in enumerate(merges)
        }

        self.pretoken_pattern: str = (
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        if self.special_tokens:
            for special_token in self.special_tokens:
                special_token_bytes = special_token.encode("utf8")
                if special_token_bytes not in self.bytes_to_id:
                    idx = len(self.vocab)
                    self.vocab[idx] = special_token_bytes
                    self.bytes_to_id[special_token_bytes] = idx
            (
                self.special_tokens.sort(key=lambda x: -len(x))
                if self.special_tokens
                else None
            )

        self.special_token_pattern: str | None = (
            "|".join(re.escape(special_token) for special_token in self.special_tokens)
            if self.special_tokens
            else None
        )

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        vocab: dict[int, bytes] = {}
        merges: list[tuple[bytes, bytes]] = []

        with open(vocab_filepath, "r") as f:
            for line in f:
                token_id, token_hex = line.split()
                vocab[int(token_id)] = bytes.fromhex(token_hex)

        with open(merges_filepath, "r") as f:
            for line in f:
                left_token_hex, right_token_hex = line.split()
                merges.append(
                    (bytes.fromhex(left_token_hex), (bytes.fromhex(right_token_hex)))
                )

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:

        results = []
        ### 按照SPEC_PAT划分
        text_segments = (
            re.split(self.special_token_pattern, text)
            if self.special_token_pattern
            else [text]
        )
        special_token_matches = (
            re.finditer(self.special_token_pattern, text)
            if self.special_token_pattern
            else None
        )

        for passage in text_segments:
            for pretoken_match in re.finditer(self.pretoken_pattern, passage):
                results += self.encode_pretoken(pretoken_match.group())

            match = next(special_token_matches, None) if special_token_matches else None
            if match:
                cur_spec_token_bytes = match.group().encode("utf8")
                cur_spec_token_id = self.bytes_to_id[cur_spec_token_bytes]
                results.append(cur_spec_token_id)

        return results

    def encode_pretoken(self, pretoken: str) -> list[int]:
        pretoken_bytes = pretoken.encode("utf8")
        if len(pretoken_bytes) == 1:
            return [self.bytes_to_id[pretoken_bytes]]
        results = []
        symbols: list[bytes] = [bytes([per_byte]) for per_byte in pretoken_bytes]

        while True:
            adjacent_pairs: list[tuple[bytes, bytes]] = [
                (first_byte, second_byte)
                for first_byte, second_byte in zip(symbols[:-1], symbols[1:])
            ]
            if not adjacent_pairs:
                break
            best_pair: tuple[bytes, bytes] = min(
                adjacent_pairs,
                key=lambda p: self.merge_rank.get(p, float("inf")),
            )
            if self.merge_rank.get(best_pair, float("inf")) == float("inf"):
                break

            symbol_index = 0
            while symbol_index < len(symbols) - 1:

                if (
                    symbols[symbol_index],
                    symbols[symbol_index + 1],
                ) == best_pair:
                    symbols[symbol_index] = best_pair[0] + best_pair[1]
                    del symbols[symbol_index + 1]
                else:
                    symbol_index += 1

        for symbol_bytes in symbols:
            results.append(self.bytes_to_id[symbol_bytes])

        return results

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pending_text = ""

        for chunk in iterable:
            cur_chunk = pending_text + chunk
            segments = (
                re.split(self.special_token_pattern, cur_chunk)
                if self.special_token_pattern
                else [cur_chunk]
            )
            special_matches = (
                re.finditer(self.special_token_pattern, cur_chunk)
                if self.special_token_pattern
                else None
            )
            for segment_idx, segment_text in enumerate(segments):
                # 只有 i 是最后一个passage的时候才会走到边界
                if segment_idx != len(segments) - 1:
                    for pretoken_match in re.finditer(
                        self.pretoken_pattern, segment_text
                    ):
                        for token_id in self.encode_pretoken(pretoken_match.group()):
                            yield token_id
                    match = next(special_matches, None) if special_matches else None
                    if match:
                        cur_spec_token_bytes = match.group().encode("utf8")
                        yield self.bytes_to_id[cur_spec_token_bytes]
                else:
                    pending_suffix_len = 0
                    if self.special_tokens:
                        for suffix_len in range(1, self.special_tokens_len_max):
                            might_be_special_token = segment_text[-suffix_len:]
                            for special_token in [
                                special_token_len_valid
                                for special_token_len_valid in self.special_tokens
                                if len(special_token_len_valid) > suffix_len
                            ]:
                                if special_token[:suffix_len] == might_be_special_token:
                                    pending_suffix_len = suffix_len
                    passage_without_special_suffix = (
                        segment_text[:-pending_suffix_len]
                        if pending_suffix_len
                        else segment_text
                    )
                    pending_pretoken: str = ""
                    for pretoken_match in re.finditer(
                        self.pretoken_pattern, passage_without_special_suffix
                    ):
                        if pending_pretoken:
                            for token_id in self.encode_pretoken(pending_pretoken):
                                yield token_id
                        pending_pretoken = pretoken_match.group()
                    pending_suffix_len += len(pending_pretoken)
                    pending_text = (
                        segment_text[-pending_suffix_len:] if pending_suffix_len else ""
                    )

        # 对buffer处理

        segments = (
            re.split(self.special_token_pattern, pending_text)
            if self.special_token_pattern
            else [pending_text]
        )
        special_matches = (
            re.finditer(self.special_token_pattern, pending_text)
            if self.special_token_pattern
            else None
        )

        for segment_text in segments:
            for pretoken_match in re.finditer(self.pretoken_pattern, segment_text):
                for token_id in self.encode_pretoken(pretoken_match.group()):
                    yield token_id
            match = next(special_matches, None) if special_matches else None
            if match:
                cur_spec_token_bytes = match.group().encode("utf8")
                yield self.bytes_to_id[cur_spec_token_bytes]

    def decode(self, ids: list[int]) -> str:
        result_bytes: bytes = bytes()
        for id in ids:
            result_bytes += self.vocab[id]
        result = result_bytes.decode("utf8", errors="replace")
        return result
