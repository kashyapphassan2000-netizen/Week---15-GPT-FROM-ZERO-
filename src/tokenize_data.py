"""
src/tokenize_data.py - Day 2 deliverable
Read cleaned JSONL, tokenize with tiktoken r50k_base, split 99/1 into train/val,
write uint16 binary shards.

    python src/tokenize_data.py                                   # blog + fineweb (if present)
    python src/tokenize_data.py --inputs data/raw/blog_posts.cleaned.jsonl

Why r50k_base and not cl100k_base: cl100k ids go up to 100,276 and don't fit in
uint16 (max 65,535), and a 100k-row embedding table at d=512 is ~51M params on its
own -- the whole budget. r50k_base (GPT-2 BPE, 50,257) fits uint16 and halves that.
"""

import argparse
import json
import pathlib
import random

import numpy as np
import tiktoken
from tqdm import tqdm

DEFAULT_INPUTS = [
    "data/raw/blog_posts.cleaned.jsonl",
    "data/raw/fineweb.jsonl",
]
OUT_DIR = pathlib.Path("data/tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    enc = tiktoken.get_encoding("r50k_base")  # GPT-2 BPE, vocab=50257
    EOT = enc.eot_token  # 50256
    dtype = np.uint16
    assert enc.n_vocab <= np.iinfo(dtype).max + 1
    print(f"[tok] vocab={enc.n_vocab}  dtype=uint16  eot={EOT}")

    texts: list[tuple[str, str]] = []
    for path in args.inputs:
        p = pathlib.Path(path)
        if not p.exists():
            print(f"[tok] skip missing {p}")
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                texts.append((rec.get("source", p.stem), rec["text"]))
        print(f"[tok] loaded {p}")
    if not texts:
        raise SystemExit("no input documents found -- run src/crawl.py + src/clean_jsonl.py first")

    # shuffle at the document level so val is a held-out mix of every source
    random.Random(args.seed).shuffle(texts)

    all_ids = []
    per_source: dict[str, int] = {}
    for source, text in tqdm(texts, desc="tokenize"):
        ids = enc.encode_ordinary(text)
        ids.append(EOT)  # document separator
        all_ids.append(np.array(ids, dtype=dtype))
        per_source[source] = per_source.get(source, 0) + len(ids)

    arr = np.concatenate(all_ids)
    n_total = len(arr)
    # spec: max(1M, 1%) -- capped at 10% so tiny corpora still keep a train split
    n_val = min(max(1_000_000, int(args.val_frac * n_total)), n_total // 10)
    print(f"[tok] total: {n_total:,}  val: {n_val:,}  train: {n_total - n_val:,}")
    for src, n in sorted(per_source.items(), key=lambda kv: -kv[1]):
        print(f"  {src:<35} {n:>12,} tokens")

    train_bin, val_bin = out_dir / "train.bin", out_dir / "val.bin"
    arr[:-n_val].tofile(train_bin)
    arr[-n_val:].tofile(val_bin)
    print(f"[tok] wrote {train_bin} ({(n_total - n_val) * 2 / 1e6:.1f} MB)")
    print(f"[tok] wrote {val_bin}   ({n_val * 2 / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
