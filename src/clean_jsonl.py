"""
src/clean_jsonl.py - Day 1 cleaning pass
Strips boilerplate lines (cookie banners, share/subscribe CTAs, copyright footers),
then site-wide boilerplate found by document frequency (e.g. the "Related content"
teaser block repeated on every anthropic.com post), collapses whitespace, drops
anything < 600 chars after cleaning.

    python src/clean_jsonl.py
"""

import argparse
import collections
import json
import pathlib
import re

INP = pathlib.Path("data/raw/blog_posts.jsonl")
OUT = pathlib.Path("data/raw/blog_posts.cleaned.jsonl")
MIN_CHARS = 600
# a prose line (>= DF_MIN_WORDS words) repeated in >= DF_MAX_DOCS documents is template text.
# Short repeated lines ("}", "import torch", "Conclusion") are code/headings and are kept.
DF_MAX_DOCS = 20
DF_MIN_WORDS = 5

BAD_LINE_PATTERNS = [
    re.compile(r"^cookie", re.I),
    re.compile(r"^subscribe", re.I),
    re.compile(r"^share this", re.I),
    re.compile(r"^© ?\d{4}"),
    re.compile(r"^all rights reserved", re.I),
    re.compile(r"^(sign up|log in|read more|read next|related (posts?|content))\b", re.I),
    re.compile(r"^get the developer newsletter", re.I),
    re.compile(r"^[×❮❯›‹»«•·|]+$"),  # carousel / UI glyph lines
]


def clean(text: str, boilerplate: frozenset = frozenset()) -> str:
    keep = []
    for ln in text.split("\n"):
        ln = ln.strip()
        if not ln or ln in boilerplate:
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

    with open(args.inp, encoding="utf-8") as fin:
        records = [json.loads(line) for line in fin]

    # pass 1: document frequency of each distinct line
    df = collections.Counter()
    for rec in records:
        df.update({ln.strip() for ln in rec["text"].split("\n") if ln.strip()})
    boilerplate = frozenset(ln for ln, n in df.items() if n >= DF_MAX_DOCS and len(ln.split()) >= DF_MIN_WORDS)
    print(f"site boilerplate: {len(boilerplate)} distinct lines repeated in >= {DF_MAX_DOCS} docs")

    # pass 2: clean
    n_in = n_out = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for rec in records:
            n_in += 1
            rec["text"] = clean(rec["text"], boilerplate)
            rec["n_chars"] = len(rec["text"])
            if rec["n_chars"] >= MIN_CHARS:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_out += 1
    print(f"cleaned {n_in} -> {n_out} records -> {args.out}")


if __name__ == "__main__":
    main()
