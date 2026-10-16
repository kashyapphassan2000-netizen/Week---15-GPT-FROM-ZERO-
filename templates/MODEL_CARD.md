---
license: mit
language: en
library_name: pytorch
tags:
  - gpt
  - from-scratch
  - pretraining
  - 50m
  - educational
---

# gpt-from-zero

A {{N_PARAMS}}-parameter GPT decoder pretrained from scratch on public AI-lab blog text
(RSS feeds from Anthropic, OpenAI, Google DeepMind, Google Research, Hugging Face, and
individual researchers' blogs) mixed with a slice of fineweb-edu.

**This is a learning artifact, not a useful LLM.** It produces grammatically
coherent but factually meaningless completions in the style of frontier-lab
announcements. Every output is machine-generated fiction. Do not use it for anything real.

## Architecture
- {{N_LAYER}} transformer blocks, {{N_HEAD}} heads, d_model={{N_EMBD}}, head_dim={{HEAD_DIM}}
- Block size: {{BLOCK_SIZE}}, tokenizer: tiktoken `r50k_base` (GPT-2 BPE, 50,257 tokens)
- RoPE positional encoding, pre-LayerNorm (no bias), GELU MLP (4x)
- FlashAttention via `F.scaled_dot_product_attention`, weight-tied embeddings
- {{N_PARAMS}} trainable parameters

## Training
- Hardware: {{HARDWARE}}
- Steps: {{STEPS}} x {{TOK_PER_STEP}} tok/step = ~{{TOKENS_SEEN}} tokens seen
- Optimizer: AdamW (b1=0.9, b2=0.95, wd={{WD}}), cosine LR {{LR}} -> {{MIN_LR}}, {{WARMUP}} warmup steps
- Grad clip 1.0, plus skip-step when the pre-clip grad norm > 5.0 (loss-spike guard)
- Best val loss: {{BEST_VAL}} (perplexity {{PPL}}) on held-out documents
- Wandb: {{WANDB_URL}}

## Data
- Mix: {{DATA_MIX}}
- Blog text: public RSS feeds, article HTML extracted with trafilatura, robots.txt respected at crawl time
- fineweb-edu (`HuggingFaceFW/fineweb-edu`, sample-10BT, ODC-By 1.0) used to reach the token budget
- Cleaning: boilerplate-line strip, whitespace collapse, sha1 dedup on the first 500 chars, min 600 chars
- License: each blog post retains its publisher's terms; this model reproduces statistical patterns, not text

## Limitations
- Hallucinates aggressively (every output is fiction)
- Topic-locked to AI-lab announcements / educational web text
- Not instruction-tuned, not RLHF'd, not safety-aligned
- 50M params is far below the scale where in-context abilities appear

## Usage
```python
import json, torch, tiktoken
from huggingface_hub import hf_hub_download
from src.model import GPT, GPTConfig          # from the GitHub repo
from src.sample import generate

ckpt = hf_hub_download("{{REPO}}", "pytorch_model.bin")
raw = json.load(open(hf_hub_download("{{REPO}}", "config.json")))
cfg = {k: v for k, v in raw.items() if k not in ("architecture", "framework", "tokenizer")}
model = GPT(GPTConfig(**cfg))
model.load_state_dict(torch.load(ckpt, map_location="cpu"))

enc = tiktoken.get_encoding("r50k_base")
x = torch.tensor([enc.encode("Today we are announcing")])
y = generate(model, x, max_new_tokens=100, temperature=0.85, top_p=0.92, device="cpu")
print(enc.decode(y[0].tolist()))
```

## Source
Built end-to-end in 7 days as part of a self-directed AI engineering roadmap.
Code: {{GITHUB_URL}}
