#!/usr/bin/env python3
"""Download RULER's public source corpora separately from experiment code."""
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent / "ruler/RULER-main/scripts/data/synthetic/json"
URLS = {
    "squad.json": ("https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",),
    "hotpotqa.json": (
        "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
        "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json",
    ),
    "english_words.json": ("https://media.githubusercontent.com/media/NVIDIA/RULER/main/scripts/data/synthetic/json/english_words.json",),
}

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    for name, urls in URLS.items():
        path = ROOT / name
        if path.is_file():
            continue
        temporary = path.with_suffix(".download")
        print(f"Downloading {name}", flush=True)
        errors = []
        for url in urls:
            try:
                with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                    import shutil
                    shutil.copyfileobj(response, output)
                break
            except OSError as error:
                errors.append(str(error))
        else:
            raise RuntimeError(f"Could not download {name}: {errors}")
        with temporary.open() as handle:
            payload = json.load(handle)
        if name == "english_words.json" and (not isinstance(payload, dict) or not all(isinstance(word, str) for word in payload.values())):
            raise ValueError("RULER english_words.json must map indices to word strings; a word-to-count dictionary is incompatible.")
        temporary.replace(path)
    essay = ROOT / "PaulGrahamEssays.json"
    if not essay.is_file():
        subprocess.run([sys.executable, str(ROOT / "download_paulgraham_essay.py")], cwd=ROOT, check=True)
        if not json.loads(essay.read_text()).get("text", "").strip():
            raise RuntimeError("Essay download produced an empty corpus; inspect the upstream URLs.")
    import nltk
    nltk.download("punkt", raise_on_error=True)
    nltk.download("punkt_tab", raise_on_error=True)
    print("RULER source corpora are ready. Generate shared samples with prepare_dataset.sh ruler.")

if __name__ == "__main__":
    main()
