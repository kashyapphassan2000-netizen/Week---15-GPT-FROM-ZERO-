#!/usr/bin/env bash
# scripts/smoke_test.sh - prove the whole pipeline on a laptop CPU in ~2 minutes
# BEFORE paying for a GPU. Uses a tiny model + a 3-posts-per-feed crawl in a temp dir.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "== unit tests";            $PY -m pytest -q
echo "== model smoke test";      $PY src/model.py
echo "== crawl (3/feed)";        $PY src/crawl.py --max-per-feed 3 --sleep 0.5 --out "$TMP/raw.jsonl" | tail -3
echo "== clean";                 $PY src/clean_jsonl.py --inp "$TMP/raw.jsonl" --out "$TMP/clean.jsonl"
echo "== tokenize";              $PY src/tokenize_data.py --inputs "$TMP/clean.jsonl" --out-dir "$TMP/tokens" | tail -3
printf 'base: configs/tiny_cpu.yaml\ndata_dir: %s/tokens\nckpt_dir: %s/ckpt\n' "$TMP" "$TMP" > "$TMP/cfg.yaml"
echo "== train 40 steps";        $PY src/train.py --config "$TMP/cfg.yaml" --no_wandb --max_steps 40 | tail -4
echo "== resume from step 20";   $PY src/train.py --config "$TMP/cfg.yaml" --no_wandb --max_steps 40 --resume "$TMP/ckpt/ckpt_step_20.pt" | grep -E "resume|done"
echo "== sample";                $PY src/sample.py --ckpt "$TMP/ckpt/ckpt_final.pt" --n 2 --max_new_tokens 20 --out "$TMP/samples.md" | tail -1
echo "== export (no push)";      $PY src/export_to_hub.py --ckpt "$TMP/ckpt/ckpt_final.pt" --out "$TMP/hub" --no_push
echo "SMOKE TEST PASSED"
