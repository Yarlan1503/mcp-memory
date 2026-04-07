#!/usr/bin/env python3
"""Download ONNX embedding model for mcp-memory.

Usage::

    python scripts/download_model.py

Downloads the model files to::

        ~/.cache/mcp-memory-v2/models/

Requires ``huggingface-hub`` (installed automatically as a dependency).
"""

import logging
import sys
from pathlib import Path

# Add src/ to path so the script works without installing the package.
_src = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_src))

from mcp_memory.embeddings import MODEL_DIR, _download_model_files  # noqa: E402

# Show download progress on stderr when run as a CLI script.
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s")


def main() -> None:
    print(f"Downloading model to {MODEL_DIR}…")
    if _download_model_files(MODEL_DIR):
        print(f"\nAll files downloaded to {MODEL_DIR}")
    else:
        print(f"\nDownload incomplete — check errors above", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
