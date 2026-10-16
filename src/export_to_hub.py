"""
src/export_to_hub.py - Day 7 deliverable
Convert a training checkpoint into pytorch_model.bin + config.json + README.md (model card)
and push it to the Hugging Face Hub.

    huggingface-cli login            # token with WRITE scope
    python src/export_to_hub.py --ckpt checkpoints/ckpt_final.pt --repo <hf-user>/gpt-from-zero \
        --wandb_url https://wandb.ai/<you>/gpt-from-zero/runs/<id> --hardware "1x A100 40GB (Vast.ai), 6h"
    python src/export_to_hub.py ... --no_push      # just write hub_export/ and inspect it
"""

import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.utils import load_model_from_ckpt  # noqa: E402

CARD_TEMPLATE = pathlib.Path(__file__).resolve().parent.parent / "templates" / "MODEL_CARD.md"


def render_card(ck: dict, n_params: int, args) -> str:
    cfg = ck.get("cfg", {})
    mcfg = ck.get("model_cfg", {})
    steps = ck.get("step", 0) + 1
    tok_per_step = cfg.get("batch_size", 0) * cfg.get("grad_accum_steps", 0) * cfg.get("block_size", 0) \
        * ck.get("world_size", args.world_size)
    best_val = ck.get("best_val", float("nan"))
    fill = {
        "{{N_PARAMS}}": f"{n_params / 1e6:.2f}M",
        "{{N_LAYER}}": str(mcfg.get("n_layer")),
        "{{N_HEAD}}": str(mcfg.get("n_head")),
        "{{N_EMBD}}": str(mcfg.get("n_embd")),
        "{{HEAD_DIM}}": str(mcfg.get("n_embd", 0) // max(1, mcfg.get("n_head", 1))),
        "{{BLOCK_SIZE}}": str(mcfg.get("block_size")),
        "{{STEPS}}": f"{steps:,}",
        "{{TOK_PER_STEP}}": f"{tok_per_step:,}",
        "{{TOKENS_SEEN}}": f"{steps * tok_per_step / 1e6:,.0f}M",
        "{{LR}}": f"{cfg.get('lr', 0):g}",
        "{{MIN_LR}}": f"{cfg.get('min_lr', 0):g}",
        "{{WARMUP}}": str(cfg.get("warmup_steps")),
        "{{WD}}": f"{cfg.get('weight_decay', 0):g}",
        "{{BEST_VAL}}": f"{best_val:.3f}",
        "{{PPL}}": f"{torch.tensor(best_val).exp().item():.1f}",
        "{{HARDWARE}}": args.hardware,
        "{{WANDB_URL}}": args.wandb_url,
        "{{REPO}}": args.repo or "<hf-username>/gpt-from-zero",
        "{{GITHUB_URL}}": args.github_url,
        "{{DATA_MIX}}": args.data_mix,
    }
    text = CARD_TEMPLATE.read_text()
    for k, v in fill.items():
        text = text.replace(k, v)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ckpt_final.pt")
    ap.add_argument("--repo", default=None, help="e.g. karthik/gpt-from-zero (required unless --no_push)")
    ap.add_argument("--out", default="hub_export")
    ap.add_argument("--wandb_url", default="<paste-run-url>")
    ap.add_argument("--hardware", default="<fill in: GPU, provider, hours>")
    ap.add_argument("--github_url", default="https://github.com/kashyapphassan2000-netizen/Week---15-GPT-FROM-ZERO-")
    ap.add_argument("--data_mix", default="<fill in: e.g. 1.9k AI-lab blog posts (6M tok) + fineweb-edu (45M tok)>")
    ap.add_argument("--world_size", type=int, default=1, help="GPUs used in training (only for old ckpts that don't record it)")
    ap.add_argument("--no_push", action="store_true")
    args = ap.parse_args()
    if not args.no_push and not args.repo:
        ap.error("--repo is required unless --no_push")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, ck = load_model_from_ckpt(args.ckpt, "cpu")

    # clean fp32 state dict (compile/DDP prefixes already stripped by the loader)
    torch.save(model.state_dict(), out / "pytorch_model.bin")
    with (out / "config.json").open("w") as f:
        json.dump({"architecture": "GPT", "framework": "pytorch", "tokenizer": "tiktoken/r50k_base",
                   **ck["model_cfg"]}, f, indent=2)
    (out / "README.md").write_text(render_card(ck, model.num_params(), args))
    print(f"[export] wrote {out}/pytorch_model.bin, config.json, README.md")

    if args.no_push:
        return
    from huggingface_hub import HfApi, create_repo
    create_repo(args.repo, exist_ok=True, repo_type="model")
    HfApi().upload_folder(folder_path=str(out), repo_id=args.repo, repo_type="model",
                          commit_message="release: gpt-from-zero 50M")
    print(f"[done] https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
