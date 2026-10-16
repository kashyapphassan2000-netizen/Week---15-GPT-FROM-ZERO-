#!/usr/bin/env bash
# scripts/launch_train.sh - one command on the rented GPU box.
#   export WANDB_API_KEY=...  HF_TOKEN=...  HF_REPO=<hf-user>/gpt-from-zero
#   bash scripts/launch_train.sh                      # 1 GPU, A100 config
#   CONFIG=configs/gpt_50m_t4.yaml NGPU=2 bash scripts/launch_train.sh
# Re-running it after a preemption resumes from the newest checkpoint automatically.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${CONFIG:-configs/gpt_50m.yaml}
NGPU=${NGPU:-1}
RUN_NAME=${RUN_NAME:-run_$(date +%m%d_%H%M)}

nvidia-smi --query-gpu=name,memory.total --format=csv
pip install -q -r requirements.txt
[ -n "${HF_TOKEN:-}" ] && huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential >/dev/null || true

# data must be on local SSD (Vast: /workspace) -- slow network disks halve tok/s
if [ ! -f data/tokens/train.bin ]; then
    echo "[launch] no token shards found -> building them (crawl + fineweb top-up + tokenize)"
    [ -f data/raw/blog_posts.jsonl ] || python src/crawl.py
    python src/clean_jsonl.py
    [ -f data/raw/fineweb.jsonl ] || python src/fetch_fineweb.py --n-docs 80000
    python src/tokenize_data.py
fi

EXTRA=()
[ -n "${HF_REPO:-}" ] && EXTRA+=(--hf_repo "$HF_REPO")
[ -z "${WANDB_API_KEY:-}" ] && { echo "[launch] WANDB_API_KEY not set -> wandb disabled"; EXTRA+=(--no_wandb); }

if [ "$NGPU" -gt 1 ]; then
    torchrun --standalone --nproc_per_node="$NGPU" src/train.py --config "$CONFIG" --resume auto --run_name "$RUN_NAME" "${EXTRA[@]}"
else
    python src/train.py --config "$CONFIG" --resume auto --run_name "$RUN_NAME" "${EXTRA[@]}"
fi
