#!/usr/bin/env python3
"""Download paraphrase-multilingual-MiniLM-L12-v2 ONNX model for mcp-memory.

Usage::

    python scripts/download_model.py

Downloads the ONNX weights and tokenizer files to::

    ~/.cache/mcp-memory-v2/models/

Requires ``huggingface-hub`` (installed automatically as a dependency).
"""

import shutil
import sys
from pathlib import Path

MODEL_DIR = Path.home() / ".cache" / "mcp-memory-v2" / "models"
MODEL_REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Tokenizer files live at the repo root — download straight to MODEL_DIR.
_ROOT_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]

# The ONNX export lives under onnx/model.onnx.  We download it via
# hf_hub_download (which caches the file) and then copy it to MODEL_DIR
# so that EmbeddingEngine can find it at a predictable path.
_ONNX_FILE = "onnx/model.onnx"


def download_model() -> None:
    """Download model files to local cache."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "ERROR: huggingface_hub not installed. Run: pip install huggingface-hub",
            file=sys.stderr,
        )
        sys.exit(1)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = 0

    # --- Root-level tokenizer files ------------------------------------
    for filename in _ROOT_FILES:
        try:
            path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename=filename,
                local_dir=MODEL_DIR,
            )
            print(f"  ✓ {filename} → {path}")
            downloaded += 1
        except Exception as e:
            print(f"  ✗ {filename}: {e}", file=sys.stderr)

    # --- ONNX model (nested under onnx/ in the repo) -------------------
    try:
        cached_onnx = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=_ONNX_FILE,
        )
        dest = MODEL_DIR / "model.onnx"
        shutil.copy2(cached_onnx, dest)
        print(f"  ✓ model.onnx → {dest}")
        downloaded += 1
    except Exception as e:
        print(f"  ✗ model.onnx: {e}", file=sys.stderr)

    # --- Summary -------------------------------------------------------
    total = len(_ROOT_FILES) + 1
    if downloaded == total:
        print(f"\nAll {downloaded} files downloaded to {MODEL_DIR}")
    else:
        print(
            f"\nDownloaded {downloaded}/{total} files",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    print(f"Downloading {MODEL_REPO} to {MODEL_DIR}...")
    download_model()
