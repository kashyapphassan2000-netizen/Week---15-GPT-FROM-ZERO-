"""Fast CPU tests -- run on the laptop before renting a GPU:  pytest -q"""
import math
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.model import GPT, GPTConfig, apply_rope, filter_logits, precompute_rope_cache  # noqa: E402
from src.utils import load_config, model_config_from  # noqa: E402

TINY = GPTConfig(block_size=32, vocab_size=101, n_layer=2, n_head=2, n_embd=32)


def test_full_config_is_about_50m():
    n = GPT(GPTConfig()).num_params()
    assert 48e6 <= n <= 55e6, f"{n / 1e6:.2f}M params -- a dimension is wrong"


def test_yaml_matches_dataclass():
    cfg = model_config_from(load_config(str(ROOT / "configs/gpt_50m.yaml")))
    assert cfg == GPTConfig()


def test_config_inheritance():
    t4 = load_config(str(ROOT / "configs/gpt_50m_t4.yaml"))
    assert t4["dtype"] == "float16" and t4["n_layer"] == 8 and t4["max_steps"] == 5000


def test_initial_loss_near_ln_vocab():
    torch.manual_seed(0)
    m = GPT(TINY)
    x = torch.randint(0, TINY.vocab_size, (4, 16))
    _, loss = m(x, torch.randint(0, TINY.vocab_size, (4, 16)))
    assert abs(loss.item() - math.log(TINY.vocab_size)) < 0.5
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())


def test_causal_no_future_leak():
    torch.manual_seed(0)
    m = GPT(TINY).eval()
    x = torch.randint(0, TINY.vocab_size, (1, 16))
    x2 = x.clone()
    x2[0, 10:] = (x2[0, 10:] + 1) % TINY.vocab_size  # change only future tokens
    l1, _ = m(x, x)  # targets given -> full-sequence logits
    l2, _ = m(x2, x2)
    assert torch.allclose(l1[0, :10], l2[0, :10], atol=1e-5)
    assert not torch.allclose(l1[0, 10:], l2[0, 10:])


def test_weight_tying():
    m = GPT(TINY)
    assert m.lm_head.weight.data_ptr() == m.tok_emb.weight.data_ptr()


def test_rope_preserves_norm_and_is_relative():
    hd, T = 16, 12
    cos, sin = precompute_rope_cache(hd, T)
    q = torch.randn(1, 1, 1, hd).expand(1, 1, T, hd)
    k = torch.randn(1, 1, 1, hd).expand(1, 1, T, hd)
    rq, rk = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    assert torch.allclose(rq.norm(dim=-1), q.norm(dim=-1), atol=1e-5)
    # <q_m, k_n> depends only on m - n
    s = lambda m_, n_: (rq[0, 0, m_] * rk[0, 0, n_]).sum()  # noqa: E731
    assert torch.allclose(s(5, 2), s(9, 6), atol=1e-4)


def test_rope_keeps_dtype_under_bf16():
    cos, sin = precompute_rope_cache(16, 8)
    x = torch.randn(1, 2, 8, 16, dtype=torch.bfloat16)
    assert apply_rope(x, cos, sin).dtype == torch.bfloat16


def test_top_p_keeps_at_least_top1_and_masks_tail():
    logits = torch.tensor([[5.0, 4.0, 0.0, -1.0, -2.0]])
    out = filter_logits(logits.clone(), top_p=0.5)
    assert torch.isfinite(out[0, 0]) and torch.isinf(out[0, 2:]).all()
    out = filter_logits(logits.clone(), top_p=0.9)
    assert torch.isfinite(out[0, :2]).all() and torch.isinf(out[0, 3:]).all()


def test_top_k():
    out = filter_logits(torch.tensor([[1.0, 3.0, 2.0, 0.0]]), top_k=2)
    assert torch.isfinite(out[0, [1, 2]]).all() and torch.isinf(out[0, [0, 3]]).all()


def test_generate_length_and_crop():
    torch.manual_seed(0)
    m = GPT(TINY)
    x = torch.randint(0, TINY.vocab_size, (1, 30))
    y = m.generate(x, max_new=10, temperature=0.8, top_p=0.9)  # crosses block_size=32
    assert y.shape == (1, 40)


def test_sample_generate_matches_api():
    from src.sample import generate
    torch.manual_seed(0)
    m = GPT(TINY)
    y = generate(m, torch.zeros(1, 1, dtype=torch.long), 5, device="cpu")
    assert y.shape == (1, 6)


@pytest.mark.parametrize("text", ["Today we are announcing Claude.", "naïve café — 日本語"])
def test_tokenizer_roundtrip_fits_uint16(text):
    tiktoken = pytest.importorskip("tiktoken")
    import numpy as np
    enc = tiktoken.get_encoding("r50k_base")
    ids = np.array(enc.encode_ordinary(text), dtype=np.uint16)
    assert enc.n_vocab <= 65536 and enc.decode(ids.tolist()) == text
