"""
scripts/push_to_hf.py - Day 7: deploy the Gradio Space.
(The model itself is pushed by src/export_to_hub.py.)

    huggingface-cli login     # WRITE-scope token
    python scripts/push_to_hf.py --space <hf-user>/gpt-from-zero --model_repo <hf-user>/gpt-from-zero
"""
import argparse
import pathlib

from huggingface_hub import HfApi, create_repo

ROOT = pathlib.Path(__file__).resolve().parent.parent
FILES = {  # local path -> path inside the Space repo
    "space/app.py": "app.py",
    "space/requirements.txt": "requirements.txt",
    "space/README.md": "README.md",
    "src/__init__.py": "src/__init__.py",
    "src/model.py": "src/model.py",
    "src/sample.py": "src/sample.py",
    "src/utils.py": "src/utils.py",
}

ap = argparse.ArgumentParser()
ap.add_argument("--space", required=True, help="<hf-user>/<space-name>")
ap.add_argument("--model_repo", required=True, help="<hf-user>/gpt-from-zero (the model the Space loads)")
args = ap.parse_args()

api = HfApi()
create_repo(args.space, repo_type="space", space_sdk="gradio", exist_ok=True)
api.add_space_variable(args.space, "MODEL_REPO", args.model_repo)
for local, remote in FILES.items():
    api.upload_file(path_or_fileobj=str(ROOT / local), path_in_repo=remote,
                    repo_id=args.space, repo_type="space", commit_message=f"deploy {remote}")
    print(f"  uploaded {remote}")
print(f"[done] https://huggingface.co/spaces/{args.space}  (first build takes ~3-5 min)")
