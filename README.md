# gpt-from-zero

A 50.9M-parameter GPT decoder (8 layers, 8 heads, d=512, RoPE) pretrained **from random init** on AI-lab blog text, built in 7 days:
crawl → tokenize → model → train → resume → sample → ship to the Hugging Face Hub plus a Gradio Space.

| | |
|---|---|
| Model | 8L · 8H · 512d · block 1024 · RoPE · SDPA/FlashAttention · weight-tied · **50.91M params** |
| Tokenizer | tiktoken `r50k_base` (GPT-2 BPE, 50,257). Not `cl100k_base`; see [Deviations](#deviations-from-the-week-13-plan) |
| Corpus | Public RSS from Anthropic, DeepMind, Google Research, HF, NVIDIA, MSR, BAIR + researcher blogs, **topped up with fineweb-edu** |
| Training | AdamW · cosine LR 3e-4→3e-5 · bf16 AMP · torch.compile · grad-norm spike skip · wandb |
| Budget | 1× A100 40GB. The 5000-step run is about 30–45 min of compute; plan for 2h including setup, so roughly US$1–3 on Vast.ai |

**Links** (fill these in after Day 7):
- HF model: `https://huggingface.co/<hf-user>/gpt-from-zero`
- HF Space: `https://huggingface.co/spaces/<hf-user>/gpt-from-zero`
- Wandb run: `<public run url>`
- Samples: [`data/samples_day6.md`](data/samples_day6.md)

---

## Status: what is done, and what still needs you

| Step | State |
|---|---|
| Crawler, cleaner, fineweb top-up, tokenizer | ✅ Written and **run here for real**. Numbers are in [Corpus stats](#corpus-stats-real-run). |
| Model | ✅ 50.91M params, initial loss 10.96 (ln 50257 = 10.82), finite grads |
| Training loop | ✅ Verified on CPU with a tiny config: loss goes down, resume, `--resume auto`, SIGTERM save, spike skip, 2-process DDP (gloo) |
| Sampling, export, Space app | ✅ Verified end-to-end against a toy checkpoint (`scripts/smoke_test.sh`) |
| Unit tests | ✅ 14 passing (`pytest -q`) |
| **The real 50M training run** | ❌ **You have to run this.** It needs a rented GPU, your wandb key and your HF token. |
| HF model repo, Space, public wandb | ❌ Depends on the training run. Each one is a single command (below). |

There is no trained model in this repo yet. Until you run Days 4–7 on a GPU, you have a verified pipeline, not a model.

---

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip wheel
pip install torch --index-url https://download.pytorch.org/whl/cpu   # laptop; the GPU image already has torch
pip install -r requirements.txt

bash scripts/smoke_test.sh     # ~2-3 min on a laptop. Do not rent a GPU until this says SMOKE TEST PASSED.
```

## Repo layout

```
configs/gpt_50m.yaml        A100 run (bf16, compile, 5000 steps x 65,536 tok)
configs/gpt_50m_t4.yaml     Kaggle/Colab T4s (fp16 + GradScaler, DDP)
configs/tiny_cpu.yaml       laptop smoke test
src/crawl.py                Day 1  RSS -> article HTML -> trafilatura -> JSONL (robots.txt, dedup, --resume)
src/clean_jsonl.py          Day 1  boilerplate strip
src/fetch_fineweb.py        Day 1  fineweb-edu top-up (the "jugaad path", but you need it, see below)
src/tokenize_data.py        Day 2  r50k_base -> uint16 train.bin / val.bin
src/model.py                Day 3  GPT + RoPE (+ `python src/model.py` smoke test)
src/train.py                Day 4-5 training, resume, spike skip, HF mirroring, DDP
src/sample.py               Day 6  top-p sampling -> data/samples_day6.md
src/export_to_hub.py        Day 7  pytorch_model.bin + config.json + model card -> HF Hub
scripts/launch_train.sh     one command on the rented box (builds data if missing, auto-resumes)
scripts/push_to_hf.py       Day 7  deploy the Gradio Space
scripts/smoke_test.sh       whole pipeline on CPU in ~2-3 min
space/                      Gradio app for the HF Space
templates/MODEL_CARD.md     model card, auto-filled from the checkpoint
notebooks/01_inspect_tokens.ipynb   Day 2 gate
```

---

## Day by day

Each day ends with its PASS gate, a commit and a tag (`git tag data-day1 && git push origin --tags`).

### Day 1: crawl
```bash
python src/crawl.py                  # ~60-70 min, polite 1.5s/request. Interrupted? add --resume
python src/clean_jsonl.py
python src/fetch_fineweb.py --n-docs 80000    # required, see "RSS alone is not enough"
wc -l data/raw/blog_posts.cleaned.jsonl
```
**PASS:** ≥500 cleaned articles, and 5 random records read as clean English.

### Day 2: tokenize
```bash
python src/tokenize_data.py          # blog + fineweb, shuffled by document, ~1% val
python scripts/verify_tokens.py      # decode roundtrip + max id < vocab
jupyter notebook notebooks/01_inspect_tokens.ipynb
```
**PASS:** both `.bin` files exist as uint16, ≥30M tokens, decoded text is readable English.

### Day 3: model
```bash
python src/model.py     # total params: 50.91M, loss ~10.9, backward OK, finite: True
pytest -q
```

### Day 4: train on the GPU
On the rented box (Vast.ai "PyTorch (cuDNN Devel)" template, ≥60GB disk, work in `/workspace`):
```bash
git clone https://github.com/kashyapphassan2000-netizen/Week---15-GPT-FROM-ZERO-.git && cd Week---15-GPT-FROM-ZERO-
export WANDB_API_KEY=...   HF_TOKEN=...   HF_REPO=<hf-user>/gpt-from-zero
bash scripts/launch_train.sh                                         # 1x A100
CONFIG=configs/gpt_50m_t4.yaml NGPU=2 bash scripts/launch_train.sh   # Kaggle 2x T4
```
Rsync `data/tokens/` up from your laptop instead of re-crawling if you already built it (`rsync -avP data/tokens gpurent:/workspace/<repo>/data/`).

What to expect on an A100 with bf16 and compile: about 180k tok/s. That is ~3 min per 500 steps, so **~30–45 min for all 5000 steps** including evals. The PDF says 4–8h; that figure is wrong for this config. On 2× T4 (fp16) expect ~40–60k tok/s, so 2–3h. Below 100k tok/s means something is wrong (compile off, fp32, or data on a network disk).
**PASS:** ≥500 steps, val loss < 5.0 at step 500, public wandb run, a checkpoint on the HF Hub.

### Day 5: resume drill
```bash
# kill training around step 300 (Ctrl-C or the provider's "stop"), then:
python src/train.py --config configs/gpt_50m.yaml --resume checkpoints/ckpt_step_250.pt --run_name day5_resume_test
# or simply re-run scripts/launch_train.sh -> --resume auto
```
Resume restores model, optimizer, GradScaler, step and the **same wandb run id**, so the curve continues instead of restarting at 10.8. A SIGTERM from a spot preemption triggers a checkpoint before exit. Checkpoints are written atomically (tmp file, then rename), and only the last 3 are kept (`keep_last_ckpts`).

Loss-spike playbook:

| Symptom | Likely cause | Fix |
|---|---|---|
| Loss jumps 3.5 → 9.0 in one step | bad batch | built in: step skipped when pre-clip grad norm > `spike_skip_norm` (5.0) |
| Slow climb over 100 steps after warmup | LR too high | lower `lr` by 30%, resume from the pre-spike ckpt |
| Flat at ~5.0, never drops | vocab mismatch / shuffled labels | `python scripts/verify_tokens.py` |
| Diverges only with bf16 | attention numerics | `dtype: float32` for a test run |
| OOM after N hours | fragmentation | halve `batch_size`, double `grad_accum_steps` |

### Day 6: sample
```bash
python src/sample.py --ckpt checkpoints/ckpt_final.pt --prompt "Today we are announcing" \
    --n 10 --temperature 0.85 --top_p 0.92 --out data/samples_day6.md
git add data/samples_day6.md
```
If the output is repetitive, the model is undertrained. Try `--temperature 0.7 --top_p 0.85` as a band-aid, but the real fix is more steps or better data.

### Day 7: ship
```bash
huggingface-cli login     # WRITE-scope token
python src/export_to_hub.py --ckpt checkpoints/ckpt_final.pt --repo <hf-user>/gpt-from-zero \
    --wandb_url <run-url> --hardware "1x A100 40GB, Vast.ai, 1.2h" \
    --data_mix "<N> AI-lab blog posts (<X>M tok) + fineweb-edu (<Y>M tok)"
python scripts/push_to_hf.py --space <hf-user>/gpt-from-zero --model_repo <hf-user>/gpt-from-zero
```
Test locally first: `python src/export_to_hub.py --no_push` and then `MODEL_REPO=hub_export python space/app.py`.
**PASS (whole week):** public model repo with weights, config and card. The Space completes a prompt from a cold start for a friend. The README links to everything. The wandb run is public.

---

## Corpus stats (real run)

Measured on 2026-10-16 with the exact code in this repo (`crawl.py` → `clean_jsonl.py` → `tokenize_data.py`):

| Source | Articles | Tokens (r50k) |
|---|---:|---:|
| huggingface.co | 878 | 2.57M |
| anthropic.com (news + engineering) | 297 | 0.44M |
| sebastianraschka.com + magazine | 155 | 0.67M |
| distill.pub | 52 | 0.36M |
| lilianweng.github.io | 52 | 0.34M |
| developer.nvidia.com | 100 | 0.28M |
| research.google | 100 | 0.18M |
| together.ai | 100 | 0.17M |
| deepmind.google | 99 | 0.12M |
| 10 smaller blogs (Gradient, Karpathy, Interconnects, AWS, Huyen, BAIR, MSR, Colah, Jalammar, Lightning) | 108 | 0.46M |
| openai.com (403 to bots; feed summaries only) | 5 | <0.01M |
| **Blog total** | **1,946** | **5.6M** (24 MB JSONL) |
| fineweb-edu, 40k docs | 40,000 | 42.8M |
| **Mix** | | **48.4M** → 47.4M train / 1.0M val |

- Crawl time: ~70 min at 1.5 s/request. Day-1 gate (≥500 articles): **PASS**. Day-2 gate (≥30M tokens): **PASS only with fineweb**.
- Cleaning: on top of the regex rules, lines with ≥5 words that repeat in ≥20 documents are dropped. That removed Anthropic's "Related content" teaser block, which was on 262 posts, and Raschka's book plugs, which were on ~90.
- Doc length: median 668 tokens, p95 ~4k.
- The default is now **80k fineweb docs (~86M tokens)**. With 48M tokens, the 328M-token run is ~6.8 epochs. With ~92M it is ~3.6 epochs, which is within the range where repeated data is still roughly as good as fresh.
- The mix is ~6–12% AI-lab blog text, so expect announcement-flavoured *topics* more than a perfect imitation of frontier-lab style. That is the trade-off for not memorising 5.6M tokens.

---

## Deviations from the Week-13 plan

The PDF had several things that would have broken or misled you. Here is what changed and why:

1. **4 of its 10 RSS URLs are dead.** `anthropic.com/rss.xml`, `ai.meta.com/blog/rss/` and `mistral.ai/news/rss.xml` return 404, and `stability.ai/news?format=rss` returns HTML, not XML. Anthropic has no official feed, so the crawler reads the community mirror `Olshansk/rss-feeds`, which lists anthropic.com URLs, and still fetches each article from anthropic.com (robots.txt allows it). I added 12 more verified feeds.
2. **OpenAI blocks automated requests (HTTP 403).** Its feed has 1,296 entries, but only one-line summaries. The crawler stops requesting openai.com after 5 refusals and does not try to get around the block. In practice you get roughly 0 OpenAI articles.
3. **RSS alone cannot meet the Day-2 gate.** Feeds only expose recent posts. The real crawl got 1,946 articles = **5.6M tokens**, against a gate of ≥30M. The PDF's "~500MB raw" corpus does not exist via RSS. A 5000-step run sees 328M tokens, which would be 50+ epochs over blog text alone, so the model would memorise it. `fetch_fineweb.py` tops the corpus up. Documents are shuffled before the train/val split, so val is a mix of both sources.
4. **The model card's "5.2B tokens seen" was wrong.** 16 × 4 × 1024 = 65,536 tok/step, so 5000 steps is **328M** tokens, not 5.2B. The card is now filled in from the checkpoint and can't drift.
5. **RoPE dtype bug.** Under bf16 autocast, the fp32 RoPE cache would promote q and k to fp32 while v stayed bf16, and `scaled_dot_product_attention` rejects mixed dtypes. The cache is now cast to the activation dtype, and a test covers it.
6. **Logging bugs in the reference loop.** The printed loss was a *sum* over 10 steps, and tok/s was underestimated by 10×. Both are averaged correctly now.
7. **The checkpoint format didn't match `sample.py`.** `train.py` saved a flat `cfg`, while `sample.py` read `ck["cfg"]["model"]` and would have raised a KeyError. Checkpoints now store `model_cfg`, and a single loader (`src/utils.load_model_from_ckpt`) handles both training checkpoints and exported weights.
8. **The spike skip broke fp16.** Skipping `scaler.step()` after `unscale_()` without calling `scaler.update()` raises on the next step. It also skipped during warmup, when large norms are normal. Both are fixed.
9. **Tiny corpora crashed the tokenizer.** `n_val = max(1M, 1%)` could be larger than the whole corpus. It is now capped at 10%.
10. **`accelerate launch` needs DDP code.** The reference loop had none. `train.py` now supports `torchrun` and `accelerate launch` (DDP, master-only eval and checkpointing, no grad sync on accumulation micro-steps).

## Honest expectations

- At 50M params and around 330M tokens, expect a final val loss of roughly 3.3–3.8 (perplexity 27–45). Samples will be grammatical, on-topic and factually fake. That is the deliverable.
- The model is not useful. What it proves is that you took a transformer from random init through training, preemption recovery, sampling and public deployment yourself. That is what you put on the CV, together with the wandb curve and the Space link.
- Cost: around US$1–3 on a Vast.ai A100 if nothing goes wrong. Budget US$10 because you will re-run at least once. Kaggle (2× T4, 30h/week) costs $0 and takes 2–3h.
- Because the run is this cheap, the best upgrade is more data plus more steps, not more hours. With ≥150M unique tokens (`fetch_fineweb.py --n-docs 150000`), `--max_steps 15000` (~1B tokens, roughly Chinchilla-optimal for 50M params) runs in ~1.5h on an A100 and gives a noticeably better model.

## License

MIT (code). Model outputs are machine-generated. Crawled blog text keeps its publishers' terms and is **not** redistributed here (`data/raw/` is gitignored).
