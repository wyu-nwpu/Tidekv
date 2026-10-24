#!/usr/bin/env python3
"""Download the public LongBench archive and extract JSONL inputs only."""
import argparse
from pathlib import Path
import shutil
import zipfile

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "longbench")
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download
    archive = hf_hub_download(repo_id="THUDM/LongBench", repo_type="dataset", filename="data.zip")
    args.output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            if member.is_dir() or not member.filename.endswith(".jsonl"):
                continue
            target = args.output / Path(member.filename).name
            if target.is_file():
                continue
            with source.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    print(args.output.resolve())

if __name__ == "__main__":
    main()
