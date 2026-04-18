import os
from typing import BinaryIO
import regex as re
from multiprocessing import Pool


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    if desired_num_chunks <= 0:
        raise ValueError("desired_num_chunks must be a positive integer")

    special_token_lenmax= 0
    for split_special_token in split_special_tokens:
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"
        if special_token_lenmax < len(split_special_token):
            special_token_lenmax = len(split_special_token)

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    split_special_num = len(split_special_tokens)
    if split_special_num == 0:
        return([0, file_size])

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        while True:
            file.seek(initial_position)
            mini_chunk = file.read(mini_chunk_size + (special_token_lenmax - 1))  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = len(mini_chunk)
            
            for i in range(split_special_num):
                cur_found_at = mini_chunk.find(split_special_tokens[i])
                if cur_found_at < found_at and cur_found_at != -1:
                    found_at = cur_found_at

            if found_at != len(mini_chunk):
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size - (special_token_lenmax - 1)

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pretokenize(input_path: str | os.PathLike, special_tokens: list[str], num_processes: int) -> dict[tuple[bytes, ...], int]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    SPEC_PAT = "|".join(re.escape(special_token) for special_token in special_tokens)
    frequency_table = {}
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, [special_token.encode("utf-8") for special_token in special_tokens])

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            passages = re.split(SPEC_PAT, chunk) if SPEC_PAT else [chunk]
            for passage in passages:
                for token in re.finditer(PAT, passage):
                    token_byte = tuple(bytes([byte_int]) for byte_int in token.group().encode("utf-8"))
                    frequency_table[token_byte] = frequency_table.get(token_byte, 0) + 1
    return frequency_table

## Usage
if __name__ == "__main__": 
    print(pretokenize("../data/test.txt",["<|endoftext|>"],64))