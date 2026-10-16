"""
src/crawl.py - Day 1 deliverable
Pulls RSS feeds, fetches each article URL, extracts main text with trafilatura,
respects robots.txt, deduplicates, writes JSONL.

    python src/crawl.py                     # all feeds -> data/raw/blog_posts.jsonl
    python src/crawl.py --max-per-feed 5    # quick smoke run
"""

import argparse
import hashlib
import json
import pathlib
import time
import urllib.robotparser
from urllib.parse import urlparse

import feedparser
import requests
import trafilatura
from tqdm import tqdm

# ---------- CONFIG ----------
# Verified 2026-10: the original Week-13 list had 4 dead URLs (anthropic.com/rss.xml,
# ai.meta.com/blog/rss/, stability.ai, mistral.ai all 404 / not XML). Anthropic has no
# official feed; the Olshansk/rss-feeds mirror lists anthropic.com URLs and we still fetch
# the articles from anthropic.com itself (robots.txt allows /news, /engineering).
FEEDS = [
    # frontier labs
    "https://raw.githubusercontent.com/Olshansk/rss-feeds/main/feeds/feed_anthropic_news.xml",
    "https://raw.githubusercontent.com/Olshansk/rss-feeds/main/feeds/feed_anthropic_engineering.xml",
    "https://openai.com/news/rss.xml",  # article pages 403 bots -> falls back to feed text
    "https://deepmind.google/blog/rss.xml",
    "https://research.google/blog/rss/",
    "https://huggingface.co/blog/feed.xml",
    "https://www.microsoft.com/en-us/research/feed/",
    "https://developer.nvidia.com/blog/feed",
    "https://www.together.ai/blog/rss.xml",
    "https://aws.amazon.com/blogs/machine-learning/feed/",
    "https://bair.berkeley.edu/blog/feed.xml",
    # individual researchers / long-form
    "https://lilianweng.github.io/index.xml",
    "https://magazine.sebastianraschka.com/feed",
    "https://sebastianraschka.com/rss_feed.xml",
    "https://jalammar.github.io/feed.xml",
    "https://karpathy.github.io/feed.xml",
    "https://distill.pub/rss.xml",
    "https://colah.github.io/rss.xml",
    "https://www.interconnects.ai/feed",
    "https://thegradient.pub/rss/",
    "https://huyenchip.com/feed.xml",
]
UA = "gpt-from-zero/0.1 (educational; contact: youremail@example.com)"
OUT_DIR = pathlib.Path("data/raw")
OUT_FILE = OUT_DIR / "blog_posts.jsonl"
MIN_CHARS = 600  # drop ultra-short posts
SLEEP_BETWEEN_REQUESTS = 1.5  # be polite

# ---------- ROBOTS ----------
_robots_cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}


def can_fetch(url: str) -> bool:
    host = urlparse(url).netloc
    if host not in _robots_cache:
        rp = urllib.robotparser.RobotFileParser()
        try:
            # fetch via requests so our UA + timeout apply (urllib's read() has neither)
            r = requests.get(f"https://{host}/robots.txt", headers={"User-Agent": UA}, timeout=15)
            if r.status_code in (401, 403):
                rp.disallow_all = True
            elif r.status_code >= 400:
                rp.allow_all = True  # no robots.txt -> allowed (standard behaviour)
            else:
                rp.parse(r.text.splitlines())
            _robots_cache[host] = rp
        except Exception:
            _robots_cache[host] = None  # unreachable robots.txt -> don't fetch
    rp = _robots_cache[host]
    return rp is not None and rp.can_fetch(UA, url)


# ---------- FETCH ----------
# Hosts that answer bots with 401/403/429 (e.g. openai.com behind Cloudflare): after
# BLOCK_AFTER consecutive refusals we stop requesting their pages -- we don't work around it.
BLOCK_AFTER = 5
_refusals: dict[str, int] = {}


def host_blocked(url: str) -> bool:
    return _refusals.get(urlparse(url).netloc, 0) >= BLOCK_AFTER


def fetch_article(url: str) -> str | None:
    if host_blocked(url) or not can_fetch(url):
        return None
    host = urlparse(url).netloc
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
        if r.status_code in (401, 403, 429):
            _refusals[host] = _refusals.get(host, 0) + 1
            if _refusals[host] == BLOCK_AFTER:
                print(f"\n  [warn] {host} refuses automated requests ({r.status_code}) -> feed text only")
            return None
        _refusals[host] = 0
        if r.status_code != 200 or not r.text:
            return None
        return trafilatura.extract(
            r.text, include_comments=False, include_tables=False, favor_recall=True
        )
    except Exception:
        return None


def feed_entry_text(entry) -> str | None:
    """Fallback: full text the publisher syndicated inside the feed itself."""
    html = ""
    if entry.get("content"):
        html = max((c.get("value", "") for c in entry["content"]), key=len)
    html = html or entry.get("summary", "")
    if not html:
        return None
    return trafilatura.extract(f"<html><body>{html}</body></html>", favor_recall=True) or None


def fetch_feed(feed_url: str):
    # feedparser's own HTTP client ignores our timeout; fetch the XML ourselves
    try:
        r = requests.get(feed_url, headers={"User-Agent": UA}, timeout=20)
        r.raise_for_status()
        return feedparser.parse(r.content)
    except Exception as e:
        print(f"  [warn] feed fetch failed: {e}")
        return feedparser.parse(b"")


# ---------- DEDUP ----------
def sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


# ---------- MAIN ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_FILE))
    ap.add_argument("--max-per-feed", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=SLEEP_BETWEEN_REQUESTS)
    ap.add_argument("--resume", action="store_true", help="append to --out, skipping URLs already in it")
    args = ap.parse_args()

    out_file = pathlib.Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    seen_hashes: set[str] = set()
    seen_urls: set[str] = set()
    if args.resume and out_file.exists():
        with out_file.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                seen_urls.add(rec["url"])
                seen_hashes.add(sha(rec["text"][:500]))
        print(f"[resume] {len(seen_urls)} articles already in {out_file}")
    n_written = 0
    per_source: dict[str, int] = {}
    with out_file.open("a" if args.resume else "w", encoding="utf-8") as fout:
        for feed_url in FEEDS:
            print(f"\n[feed] {feed_url}")
            feed = fetch_feed(feed_url)
            if feed.bozo and not feed.entries:
                print(f"  [warn] no entries ({feed.get('bozo_exception', 'unknown error')})")
            entries = feed.entries[: args.max_per_feed] if args.max_per_feed else feed.entries
            for entry in tqdm(entries, desc=urlparse(feed_url).netloc):
                url = entry.get("link")
                title = entry.get("title", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                fetched = not host_blocked(url)
                text = fetch_article(url)
                if fetched:
                    time.sleep(args.sleep)
                if not text or len(text) < MIN_CHARS:
                    text = feed_entry_text(entry)
                if not text or len(text) < MIN_CHARS:
                    continue
                h = sha(text[:500])
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                source = urlparse(url).netloc  # article host (feed may be a mirror)
                record = {
                    "source": source,
                    "url": url,
                    "title": title,
                    "text": text,
                    "n_chars": len(text),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1
                per_source[source] = per_source.get(source, 0) + 1

    print("\n[per-source]")
    for src, n in sorted(per_source.items(), key=lambda kv: -kv[1]):
        print(f"  {src:<35} {n}")
    print(f"\n[done] wrote {n_written} articles to {out_file}")


if __name__ == "__main__":
    main()
