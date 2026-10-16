"""
src/utils.py - shared helpers: config loading, device/dtype, checkpoint I/O.
"""

import os
import pathlib
import re
from dataclasses import fields

import torch
import yaml

from src.model import GPT, GPTConfig


# ---------------- config ----------------
def load_config(path: str) -> dict:
    """YAML loader with one level of `base:` inheritance (child keys win)."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        merged = load_config(base)
        merged.update(cfg)
        cfg = merged
    return cfg


def model_config_from(cfg: dict) -> GPTConfig:
    keys = {f.name for f in fields(GPTConfig)}
    return GPTConfig(**{k: v for k, v in cfg.items() if k in keys})


# ---------------- device ----------------
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


PTDTYPE = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def resolve_dtype(name: str, device: str) -> torch.dtype:
    dt = PTDTYPE[name]
    if device != "cuda" and dt != torch.float32:
        return torch.float32  # AMP is a CUDA thing here; keep CPU/MPS runs in fp32
    if dt == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("[warn] bf16 not supported on this GPU (T4/V100?) -> using float16 + GradScaler")
        return torch.float16
    return dt


# ---------------- checkpoints ----------------
def strip_compile_prefix(sd: dict) -> dict:
    return {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}


def atomic_save(obj, path: pathlib.Path):
    """Write to tmp then rename: a spot-instance kill mid-save can't corrupt the file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


_STEP_RE = re.compile(r"ckpt_step_(\d+)\.pt$")


def list_step_ckpts(ckpt_dir: pathlib.Path) -> list[pathlib.Path]:
    found = [(int(m.group(1)), p) for p in ckpt_dir.glob("ckpt_step_*.pt") if (m := _STEP_RE.search(p.name))]
    return [p for _, p in sorted(found)]


def find_latest_ckpt(ckpt_dir: pathlib.Path) -> pathlib.Path | None:
    latest = ckpt_dir / "ckpt_latest.pt"
    if latest.exists():
        return latest
    steps = list_step_ckpts(ckpt_dir)
    return steps[-1] if steps else None


def prune_ckpts(ckpt_dir: pathlib.Path, keep: int):
    if keep and keep > 0:
        for p in list_step_ckpts(ckpt_dir)[:-keep]:
            p.unlink(missing_ok=True)


def load_model_from_ckpt(path: str, device: str = "cpu") -> tuple[GPT, dict]:
    """Load a training checkpoint (ckpt_*.pt) OR an exported pytorch_model.bin + config.json."""
    path = pathlib.Path(path)
    ck = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ck, dict) and "model" in ck:  # training checkpoint
        mcfg = ck.get("model_cfg") or model_config_from(ck["cfg"]).__dict__
        sd = ck["model"]
    else:  # exported state dict, config.json next to it
        import json
        raw = json.loads((path.parent / "config.json").read_text())
        mcfg = {k: v for k, v in raw.items() if k not in ("architecture", "framework")}
        sd = ck
    cfg = GPTConfig(**{k: v for k, v in mcfg.items() if k in {f.name for f in fields(GPTConfig)}})
    model = GPT(cfg)
    model.load_state_dict(strip_compile_prefix(sd))
    return model.to(device).eval(), ck if isinstance(ck, dict) else {}
