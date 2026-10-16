"""
Hugging Face Space (Gradio, CPU basic): prompt -> completion from the gpt-from-zero model.
Deployed by scripts/push_to_hf.py space, which uploads this file + src/{model,sample,utils}.py.
Local test:  MODEL_REPO=hub_export python space/app.py   (or MODEL_REPO=<hf-user>/gpt-from-zero)
"""
import os
import pathlib
import sys

import gradio as gr
import tiktoken
import torch
from huggingface_hub import hf_hub_download

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # on the Space: ./src/ sits next to app.py
sys.path.insert(0, str(HERE.parent))  # local run from the GitHub repo
from src.sample import generate  # noqa: E402
from src.utils import load_model_from_ckpt  # noqa: E402

REPO = os.environ.get("MODEL_REPO", "<your-hf-username>/gpt-from-zero")
torch.set_num_threads(max(1, os.cpu_count() or 1))

if pathlib.Path(REPO).is_dir():  # local test against a hub_export/ folder
    ckpt_path = str(pathlib.Path(REPO) / "pytorch_model.bin")
else:
    ckpt_path = hf_hub_download(REPO, "pytorch_model.bin")
    hf_hub_download(REPO, "config.json")  # lands next to the weights; the loader reads it from there
model, _ = load_model_from_ckpt(ckpt_path, "cpu")
enc = tiktoken.get_encoding("r50k_base")


def complete(prompt, max_new=120, temperature=0.85, top_p=0.92):
    prompt = (prompt or "").strip() or "Today we are announcing"
    ids = enc.encode(prompt)[-(model.cfg.block_size - 1):]
    x = torch.tensor([ids], dtype=torch.long)
    y = generate(model, x, max_new_tokens=int(max_new), temperature=float(temperature),
                 top_p=float(top_p), eos_token=enc.eot_token, device="cpu")
    return enc.decode([t for t in y[0].tolist() if t != enc.eot_token])


demo = gr.Interface(
    fn=complete,
    inputs=[
        gr.Textbox(label="Prompt", value="Today we are announcing"),
        gr.Slider(20, 300, value=120, step=10, label="max new tokens"),
        gr.Slider(0.1, 1.5, value=0.85, step=0.05, label="temperature"),
        gr.Slider(0.5, 1.0, value=0.92, step=0.01, label="top_p"),
    ],
    outputs=gr.Textbox(label="Completion (machine-generated)", lines=10),
    title="gpt-from-zero (50M, from scratch)",
    description="A from-scratch 50M GPT trained on AI-lab announcement text. "
                "Outputs are grammatically coherent but factually fake. Educational artifact.",
)

if __name__ == "__main__":
    demo.launch()
