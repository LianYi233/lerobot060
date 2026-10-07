#!/usr/bin/env python3
"""Download a pinned public LeRobot PI05 DROID checkpoint without loading ML or robot code."""

import argparse
import hashlib
import json
import os
from pathlib import Path

REPO_ID = "lerobot/pi05_droid"
REVISION = "72824c0a93f00ce5bb8bedb7feb58953ba1da364"
# Official file details: https://huggingface.co/lerobot/pi05_droid/blob/main/model.safetensors
WEIGHTS_SHA256 = "684c8b2033dcfacee9c6a83f38810ecc77df2c81aeb171bb87319da01fafcfaa"
FILES = ("config.json", "policy_preprocessor.json", "policy_postprocessor.json", "model.safetensors")


def check_download(root: Path) -> None:
    for name in FILES:
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty file: {path}")
        if path.suffix == ".json":
            with path.open(encoding="utf-8") as stream:
                if not isinstance(json.load(stream), dict):
                    raise ValueError(f"Expected a JSON object: {path}")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    for key, expected in {
        "type": "pi05",
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
        "max_state_dim": 32,
        "max_action_dim": 32,
    }.items():
        if config.get(key) != expected:
            raise ValueError(f"Unexpected {key}: {config.get(key)!r}; expected {expected!r}")
    print("Checking the complete model.safetensors SHA256 (reads ~16.6 GB)...", flush=True)
    digest = hashlib.sha256()
    with (root / "model.safetensors").open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != WEIGHTS_SHA256:
        raise ValueError(
            f"DROID weights SHA256 mismatch: expected {WEIGHTS_SHA256}, got {actual}. "
            "Keep this run stopped; check the downloaded file/source before training."
        )


def download(root: Path, endpoint: str) -> None:
    # Set these before importing huggingface_hub; no environment packages are changed.
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "20")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    from huggingface_hub import hf_hub_download

    for name in FILES:
        print(f"Downloading {REPO_ID}@{REVISION}: {name}", flush=True)
        hf_hub_download(
            repo_id=REPO_ID,
            revision=REVISION,
            filename=name,
            local_dir=root,
            endpoint=endpoint,
            token=False,  # Public files; never send stored account tokens to a mirror.
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    parser.add_argument(
        "--check_only", action="store_true", help="Verify existing files without network access"
    )
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    try:
        if not args.check_only:
            download(root, args.endpoint)
        check_download(root)
    except Exception as exc:
        parser.exit(1, f"DROID_DOWNLOAD_FAILED: {exc}\n")
    print(
        f"DROID_FILES_OK: {root}; pinned weights verified; model loading/GPU training not tested.", flush=True
    )


if __name__ == "__main__":
    main()
