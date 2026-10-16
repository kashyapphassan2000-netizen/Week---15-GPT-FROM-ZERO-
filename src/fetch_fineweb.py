"""
src/fetch_fineweb.py - Day 1 JUGAAD PATH / corpus top-up
Streams documents from HuggingFaceFW/fineweb-edu (sample-10BT, ODC-By licensed)
into the same JSONL schema the crawler writes.

Why this exists: RSS feeds only expose each blog's most recent ~10-50 posts, so the
crawl yields far less than the plan assumed (measured 2026-10: 1,946 articles = 5.6M
tokens from 21 feeds). The Day-2 gate wants >=30M tokens, and 5000 steps x 65,536 tok/step
= 328M tokens seen, i.e. ~58 epochs over blog-only data -> memorisation. fineweb-edu fixes both.

    python src/fetch_fineweb.py --n-docs 80000      # ~86M tokens (measured: 40k docs = 42.8M)
"""

import argparse
import json
import pathlib

from tqdm import tqdm

OUT = pathlib.Path("data/raw/fineweb.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=80_000)
    ap.add_argument("--min-chars", type=int, default=600)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    from datasets import load_dataset  # heavy import, keep it local

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f, tqdm(total=args.n_docs, desc="fineweb-edu") as bar:
        for r in ds:
            text = r["text"]
            if len(text) < args.min_chars:
                continue
            f.write(json.dumps({
                "source": "fineweb-edu",
                "url": r.get("url", ""),
                "title": "",
                "text": text,
                "n_chars": len(text),
            }, ensure_ascii=False) + "\n")
            n += 1
            bar.update(1)
            if n >= args.n_docs:
                break
    print(f"[done] wrote {n} docs to {out}")


if __name__ == "__main__":
    main()
