"""
src/model.py - Day 3 deliverable
50M-parameter GPT decoder with RoPE positional embeddings.
Adapted from nanoGPT (MIT) with RoPE bolted on.

    python src/model.py     # smoke test: ~50.9M params, initial loss ~ln(50257)=10.82
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = False


# -------------------------- RoPE --------------------------
def precompute_rope_cache(head_dim, max_seq, base=10_000.0, device=None, dtype=torch.float32):
    """Returns (cos, sin) of shape (max_seq, head_dim/2)."""
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    seq = torch.arange(max_seq, device=device, dtype=dtype)
    freqs = torch.outer(seq, theta)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    """x: (B, n_head, T, head_dim). Rotate interleaved feature pairs."""
    T = x.size(-2)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    # cast the cache to x's dtype: under bf16 autocast q/k are bf16 while the cache
    # is fp32, and a dtype mismatch between q/k and v breaks scaled_dot_product_attention
    cos = cos[:T].to(x.dtype)[None, None]
    sin = sin[:T].to(x.dtype)[None, None]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.stack((rx1, rx2), dim=-1).flatten(-2)


# ------------------ Attention + MLP + Block ------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.out = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout

    def forward(self, x, rope_cos, rope_sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)
        # PyTorch 2.x picks FlashAttention when available
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(y)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x, rope_cos, rope_sin):
        x = x + self.attn(self.ln1(x), rope_cos, rope_sin)
        x = x + self.mlp(self.ln2(x))
        return x


# -------------------------- GPT --------------------------
class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        cos, sin = precompute_rope_cache(cfg.n_embd // cfg.n_head, cfg.block_size)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # scaled init for residual projections (GPT-2 paper)
        for n, p in self.named_parameters():
            if n.endswith("out.weight") or n.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())  # tied weight counted once

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"seq len {T} > block_size {self.cfg.block_size}"
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        x = self.ln_f(x)
        if targets is None:
            # inference: only the last position's logits are needed
            return self.lm_head(x[:, [-1], :]), None
        logits = self.lm_head(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(), targets.view(-1), ignore_index=-1
        )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=1.0, top_k=None, top_p=None, eos_token=None):
        self.eval()
        for _ in range(max_new):
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :].float() / max(temperature, 1e-6)
            logits = filter_logits(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
            if eos_token is not None and idx.size(0) == 1 and nxt.item() == eos_token:
                break
        return idx


def filter_logits(logits, top_k=None, top_p=None):
    """Top-k then top-p (nucleus) filtering on (B, V) logits."""
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[:, [-1]], -float("inf"))
    if top_p is not None and 0.0 < top_p < 1.0:
        sl, si = torch.sort(logits, descending=True)
        cum = F.softmax(sl, dim=-1).cumsum(dim=-1)
        mask = cum > top_p
        mask[..., 1:] = mask[..., :-1].clone()  # shift right: keep the token that crosses p
        mask[..., 0] = False  # always keep the top-1
        sl = sl.masked_fill(mask, -float("inf"))
        logits = torch.full_like(logits, -float("inf")).scatter(-1, si, sl)
    return logits


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = GPTConfig()
    m = GPT(cfg)
    print(f"total params: {m.num_params() / 1e6:.2f}M")

    x = torch.randint(0, cfg.vocab_size, (2, 64))
    y = torch.randint(0, cfg.vocab_size, (2, 64))
    logits, loss = m(x, y)
    print("logits:", tuple(logits.shape), "loss:", round(loss.item(), 3),
          f"(expected ~ln({cfg.vocab_size}) = {math.log(cfg.vocab_size):.2f})")
    loss.backward()
    grads_finite = all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    print("backward OK, grad mean:", m.tok_emb.weight.grad.abs().mean().item(), "finite:", grads_finite)
