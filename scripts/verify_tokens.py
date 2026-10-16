"""Day 2 roundtrip check: decode a slice of train.bin and eyeball it."""
import sys

import numpy as np
import tiktoken

path = sys.argv[1] if len(sys.argv) > 1 else "data/tokens/train.bin"
arr = np.memmap(path, dtype=np.uint16, mode="r")
enc = tiktoken.get_encoding("r50k_base")
print("shape:", arr.shape, " max id:", int(arr.max()))
assert int(arr.max()) < enc.n_vocab, "token id out of vocab range -- wrong tokenizer/dtype"
start = min(10_000, max(0, len(arr) - 200))
print("decoded sample:")
print(enc.decode(arr[start:start + 200].tolist()))
