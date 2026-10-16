"""
src/clean_jsonl.py - Day 1 cleaning pass
Strips boilerplate lines (cookie banners, share/subscribe CTAs, copyright footers),
collapses whitespace, drops anything < 600 chars after cleaning.

    python src/clean_jsonl.py
"""

import argparse
import json
import pathlib
import re

INP = pathlib.Path("data/raw/blog_posts.jsonl")
OUT = pathlib.Path("data/raw/blog_posts.cleaned.jsonl")
MIN_CHARS = 600

BAD_LINE_PATTERNS = [
    re.compile(r"^cookie", re.I),
    re.compile(r"^subscribe", re.I),
    re.compile(r"^share this", re.I),
    re.compile(r"^© ?\d{4}"),
    re.compile(r"^all rights reserved", re.I),
    re.compile(r"^(sign up|log in|read more|related posts?)\b", re.I),
]


def clean(text: str) -> str:
    keep = []
    for ln in text.split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        if any(p.search(ln) for p in BAD_LINE_PATTERNS):
            continue
        keep.append(ln)
    out = "\n".join(keep)
    return re.sub(r"[ \t]+", " ", out)  # collapse multi-spaces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default=str(INP))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    n_in = n_out = 0
    with open(args.inp, encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            n_in += 1
            rec = json.loads(line)
            rec["text"] = clean(rec["text"])
            rec["n_chars"] = len(rec["text"])
            if rec["n_chars"] >= MIN_CHARS:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_out += 1
    print(f"cleaned {n_in} -> {n_out} records -> {args.out}")


if __name__ == "__main__":
    main()
