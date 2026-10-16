"""
src/train.py - Day 4 / 5 deliverable
Train the 50M GPT with AdamW + cosine schedule + AMP + wandb.

Single GPU:
    python src/train.py --config configs/gpt_50m.yaml --run_name day4_run1
Multi GPU (Kaggle 2x T4, 4x T4, ...):
    torchrun --standalone --nproc_per_node=2 src/train.py --config configs/gpt_50m_t4.yaml
Resume (Day 5):
    python src/train.py --resume checkpoints/ckpt_step_250.pt --run_name day5_resume_test
    python src/train.py --resume auto          # newest checkpoint in ckpt_dir
Laptop smoke test:
    python src/train.py --config configs/tiny_cpu.yaml --no_wandb
"""

import argparse
import math
import os
import pathlib
import signal
import sys
import threading
import time
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.model import GPT  # noqa: E402
from src.utils import (  # noqa: E402
    atomic_save, find_latest_ckpt, get_device, load_config, model_config_from,
    prune_ckpts, resolve_dtype, strip_compile_prefix,
)

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/gpt_50m.yaml")
parser.add_argument("--resume", type=str, default=None,
                    help="path to checkpoint .pt to resume from, or 'auto' for the newest in ckpt_dir")
parser.add_argument("--wandb_project", default="gpt-from-zero")
parser.add_argument("--run_name", default=None)
parser.add_argument("--no_wandb", action="store_true")
parser.add_argument("--hf_repo", default=os.environ.get("HF_REPO"),
                    help="e.g. <hf-username>/gpt-from-zero -- mirror checkpoints to the Hub")
parser.add_argument("--max_steps", type=int, default=None, help="override config max_steps")
args = parser.parse_args()

cfg = load_config(args.config)
if args.max_steps is not None:
    cfg["max_steps"] = args.max_steps

# ---- distributed (torchrun / accelerate launch set RANK etc.)
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    from torch.distributed import destroy_process_group, init_process_group
    from torch.nn.parallel import DistributedDataParallel as DDP
    rank, local_rank, world = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    if torch.cuda.is_available():
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        init_process_group(backend="nccl")
    else:
        device = "cpu"
        init_process_group(backend="gloo")
else:
    rank, local_rank, world = 0, 0, 1
    device = get_device()
master = rank == 0
device_type = "cuda" if device.startswith("cuda") else device


def log(*a):
    if master:
        print(*a, flush=True)


ptdtype = resolve_dtype(cfg["dtype"], device_type)
ctx = torch.autocast(device_type="cuda", dtype=ptdtype) if device_type == "cuda" and ptdtype != torch.float32 \
    else nullcontext()

torch.manual_seed(cfg["seed"] + rank)
if device_type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ---- data
data_dir = pathlib.Path(cfg["data_dir"])
B, T, ACC = cfg["batch_size"], cfg["block_size"], cfg["grad_accum_steps"]


def get_batch(split):
    # re-open the memmap every batch: avoids the slow memory creep of a long-lived np.memmap
    fname = cfg["train_bin"] if split == "train" else cfg["val_bin"]
    data = np.memmap(data_dir / fname, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - T - 1, (B,))
    x = torch.stack([torch.from_numpy(data[i:i + T].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + T].astype(np.int64)) for i in ix])
    if device_type == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


# ---- model
mcfg = model_config_from(cfg)
model = GPT(mcfg).to(device)
log(f"params: {model.num_params() / 1e6:.2f}M  device={device} x{world}  dtype={ptdtype}")
tok_per_step = B * ACC * T * world
log(f"tokens/step: {tok_per_step:,}  total tokens: {tok_per_step * cfg['max_steps'] / 1e6:.0f}M")

# Decay everything except biases and layernorms (all <2-D params)
decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
optimizer = torch.optim.AdamW(
    [{"params": decay, "weight_decay": cfg["weight_decay"]},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]),
    fused=(device_type == "cuda"),
)
scaler = torch.amp.GradScaler("cuda", enabled=(ptdtype == torch.float16))


def get_lr(step):
    if step < cfg["warmup_steps"]:
        return cfg["lr"] * (step + 1) / cfg["warmup_steps"]
    if step > cfg["max_steps"]:
        return cfg["min_lr"]
    ratio = (step - cfg["warmup_steps"]) / max(1, cfg["max_steps"] - cfg["warmup_steps"])
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg["min_lr"] + coeff * (cfg["lr"] - cfg["min_lr"])


# ---- resume (Day 5)
ckpt_dir = pathlib.Path(cfg.get("ckpt_dir", "checkpoints"))
ckpt_dir.mkdir(parents=True, exist_ok=True)
start_step, best_val, wandb_id = 0, float("inf"), None
if args.resume:
    path = find_latest_ckpt(ckpt_dir) if args.resume == "auto" else pathlib.Path(args.resume)
    if path is None:
        log("[resume] 'auto' but no checkpoint found -> fresh start")
    else:
        ck = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(strip_compile_prefix(ck["model"]))
        optimizer.load_state_dict(ck["optimizer"])
        if ck.get("scaler") and scaler.is_enabled():
            scaler.load_state_dict(ck["scaler"])
        start_step = ck["step"] + 1
        best_val = ck.get("best_val", float("inf"))
        wandb_id = ck.get("wandb_id")
        log(f"[resume] loaded {path} at step {ck['step']}, best_val={best_val:.4f}")
        del ck

raw_model = model
if cfg.get("compile", False) and device_type == "cuda":
    log("torch.compile()...")
    model = torch.compile(model)
eval_model = model  # master-only eval must bypass the DDP wrapper (its forward is a collective)
if ddp:
    # rope buffers are deterministic -> no need to broadcast them every forward
    model = DDP(model, device_ids=[local_rank] if device_type == "cuda" else None, broadcast_buffers=False)

# ---- wandb
use_wandb = master and not args.no_wandb
if use_wandb:
    import wandb
    run = wandb.init(project=args.wandb_project, name=args.run_name, config=cfg,
                     id=wandb_id, resume="allow")  # same run id -> resumed curve is continuous
    wandb_id = run.id

# ---- HF Hub mirroring (cheap insurance against a vanished VM)
hf_api, hf_threads = None, []
if master and args.hf_repo:
    from huggingface_hub import HfApi, create_repo
    hf_api = HfApi()
    create_repo(args.hf_repo, exist_ok=True, repo_type="model")
    log(f"[hf] mirroring checkpoints to https://huggingface.co/{args.hf_repo}")


def hf_upload(local: pathlib.Path, step: int):
    """Upload in a background thread so training doesn't stall on a 600MB push."""
    if hf_api is None:
        return

    def _up():
        try:
            hf_api.upload_file(path_or_fileobj=str(local), path_in_repo="checkpoints/ckpt_latest.pt",
                               repo_id=args.hf_repo, repo_type="model",
                               commit_message=f"checkpoint step {step}")
            print(f"  [hf] uploaded step {step}", flush=True)
        except Exception as e:  # never kill training over an upload
            print(f"  [hf] upload failed at step {step}: {e}", flush=True)

    t = threading.Thread(target=_up, daemon=False)
    t.start()
    hf_threads.append(t)


@torch.no_grad()
def estimate_loss():
    out = {}
    eval_model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(cfg["eval_iters"])
        for k in range(cfg["eval_iters"]):
            x, y = get_batch(split)
            with ctx:
                _, loss = eval_model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    eval_model.train()
    return out


def save_ckpt(step, name=None):
    obj = {
        "step": step,
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "best_val": best_val,
        "cfg": cfg,
        "model_cfg": asdict(mcfg),
        "wandb_id": wandb_id,
        "world_size": world,
    }
    p = ckpt_dir / (name or f"ckpt_step_{step}.pt")
    atomic_save(obj, p)
    if name is None:
        atomic_save(obj, ckpt_dir / "ckpt_latest.pt")
        prune_ckpts(ckpt_dir, cfg.get("keep_last_ckpts", 0))
    print(f"  [save] {p}", flush=True)
    return p


# ---- preemption: spot/interruptible hosts send SIGTERM before killing the box
stop_requested = False


def _on_sigterm(signum, frame):
    global stop_requested
    stop_requested = True
    log(f"[signal] {signal.Signals(signum).name} -> checkpoint at end of this step, then exit")


signal.signal(signal.SIGTERM, _on_sigterm)

# ---- train loop
model.train()
step = start_step
x, y = get_batch("train")
t0 = time.time()
loss_acc, n_logged, n_skipped = 0.0, 0, 0
upload_every = cfg.get("hf_upload_interval", 0)

while step < cfg["max_steps"]:
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    step_loss = 0.0
    for micro in range(ACC):
        if ddp:  # only all-reduce grads on the last micro-step
            model.require_backward_grad_sync = micro == ACC - 1
        with ctx:
            _, loss = model(x, y)
            loss = loss / ACC
        x, y = get_batch("train")  # prefetch next while the GPU is busy
        scaler.scale(loss).backward()
        step_loss += loss.item()

    scaler.unscale_(optimizer)
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"] or float("inf"))
    # Day 5 anti-spike: skip this update when the pre-clip grad norm explodes (after warmup).
    # With fp16 a non-finite norm is normal GradScaler behaviour -> scaler.step skips it itself.
    spike = cfg.get("spike_skip_norm")
    if spike and step >= cfg["warmup_steps"] and torch.isfinite(total_norm) and total_norm.item() > spike:
        log(f"[skip] step {step}  bad grad norm {total_norm.item():.2f}")
        n_skipped += 1
        scaler.update()  # resets the unscale_ bookkeeping
    else:
        scaler.step(optimizer)
        scaler.update()
    optimizer.zero_grad(set_to_none=True)

    loss_acc += step_loss
    n_logged += 1
    if step % cfg["log_interval"] == 0 and master:
        dt = time.time() - t0
        tps = tok_per_step * n_logged / max(dt, 1e-9)
        avg = loss_acc / n_logged
        print(f"step {step:>5}  loss {avg:.4f}  lr {lr:.2e}  norm {total_norm.item():.2f}  tok/s {tps:,.0f}",
              flush=True)
        if use_wandb:
            wandb.log({"loss/train_step": avg, "lr": lr, "grad_norm": total_norm.item(),
                       "tok/s": tps, "skipped_steps": n_skipped}, step=step)
        loss_acc, n_logged, t0 = 0.0, 0, time.time()

    if step % cfg["eval_interval"] == 0 and step > 0 and master:
        ev = estimate_loss()
        print(f"  [eval] step {step}  train {ev['train']:.4f}  val {ev['val']:.4f}", flush=True)
        if use_wandb:
            wandb.log({"loss/val": ev["val"], "loss/train_eval": ev["train"]}, step=step)
        if ev["val"] < best_val:
            best_val = ev["val"]

    if master and ((step % cfg["save_interval"] == 0 and step > 0) or stop_requested):
        p = save_ckpt(step)
        if upload_every and step % upload_every == 0:
            hf_upload(p, step)

    if stop_requested:
        break
    step += 1

if master:
    if not stop_requested:
        ev = estimate_loss()
        best_val = min(best_val, ev["val"])
        log(f"  [eval] final  train {ev['train']:.4f}  val {ev['val']:.4f}")
        if use_wandb:
            wandb.log({"loss/val": ev["val"], "loss/train_eval": ev["train"]}, step=step)
        final = ckpt_dir / "ckpt_final.pt"
        save_ckpt(step - 1, name="ckpt_final.pt")
        hf_upload(final, step)
    for t in hf_threads:
        t.join()
    log(f"[done] best_val={best_val:.4f}  skipped_steps={n_skipped}")
    if use_wandb:
        wandb.finish()
if ddp:
    torch.distributed.barrier()
    destroy_process_group()
