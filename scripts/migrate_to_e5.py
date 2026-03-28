#!/usr/bin/env python3
"""Migrate embedding model to intfloat/multilingual-e5-small (ONNX).

Usage::

    uv run python scripts/migrate_to_e5.py

Steps:
    1. Backup current model files to ~/.cache/mcp-memory-v2/models-backup-paraphrase/
    2. Download intfloat/multilingual-e5-small ONNX (FP32) + tokenizer
    3. Verify model loads and produces correct output

After running this script, the code changes in embeddings.py and server.py
must be applied for the new model to work with query/passage prefixes.

To rollback::

    rm -rf ~/.cache/mcp-memory-v2/models
    mv ~/.cache/mcp-memory-v2/models-backup-paraphrase ~/.cache/mcp-memory-v2/models
"""

import shutil
import sys
from pathlib import Path

MODEL_DIR = Path.home() / ".cache" / "mcp-memory-v2" / "models"
BACKUP_DIR = MODEL_DIR.parent / "models-backup-paraphrase"
NEW_REPO = "intfloat/multilingual-e5-small"

# ONNX model file: repo path -> local filename
_ONNX_FILES: list[tuple[str, str]] = [
    ("onnx/model.onnx", "model.onnx"),  # FP32 (470 MB)
]

# Tokenizer files from repo root -> downloaded to MODEL_DIR directly
_ROOT_FILES: list[str] = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]


def backup_current() -> None:
    """Backup existing model files."""
    if not MODEL_DIR.exists():
        print("  No existing model directory — skipping backup")
        return

    if BACKUP_DIR.exists():
        print(f"  Backup already exists at {BACKUP_DIR} — skipping")
        return

    print(f"  Backing up current model to {BACKUP_DIR}...")
    shutil.copytree(MODEL_DIR, BACKUP_DIR)
    print(f"  ✓ Backup complete")


def download_new_model() -> bool:
    """Download e5-small ONNX + tokenizer files."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  ERROR: huggingface_hub not available.")
        print("  Run: uv pip install huggingface-hub")
        return False

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0

    # --- ONNX model (from onnx/ subdirectory) -----------------------
    print(f"  Downloading ONNX model from {NEW_REPO}...")
    for repo_file, local_name in _ONNX_FILES:
        try:
            cached = hf_hub_download(repo_id=NEW_REPO, filename=repo_file)
            dest = MODEL_DIR / local_name
            shutil.copy2(cached, dest)
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  ✓ {local_name} ({size_mb:.1f} MB)")
            downloaded += 1
        except Exception as e:
            print(f"  ✗ {local_name}: {e}")
            return False

    # --- Tokenizer files (from repo root) ---------------------------
    print(f"  Downloading tokenizer files...")
    for filename in _ROOT_FILES:
        try:
            path = hf_hub_download(
                repo_id=NEW_REPO,
                filename=filename,
                local_dir=MODEL_DIR,
            )
            size_mb = Path(path).stat().st_size / (1024 * 1024)
            print(f"  ✓ {filename} ({size_mb:.1f} MB)")
            downloaded += 1
        except Exception as e:
            print(f"  ✗ {filename}: {e}")
            return False

    total = len(_ONNX_FILES) + len(_ROOT_FILES)
    print(f"  Downloaded {downloaded}/{total} files to {MODEL_DIR}")
    return downloaded == total


def verify_model() -> bool:
    """Verify the model loads and produces correct output."""
    print("  Verifying model...")
    try:
        # Import from the project source
        import numpy as np

        src_dir = Path(__file__).parent.parent / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        from mcp_memory.embeddings import EmbeddingEngine, DIMENSION

        # Reset singleton to force reload with new model
        EmbeddingEngine.reset()
        engine = EmbeddingEngine()

        if not engine.available:
            print("  ✗ Model failed to load — check files in", MODEL_DIR)
            return False

        # Test encode with e5-small query prefix
        result = engine.encode(["query: hola mundo"])
        if result.shape != (1, DIMENSION):
            print(f"  ✗ Unexpected shape: {result.shape} (expected (1, {DIMENSION}))")
            return False

        # Verify L2 normalization
        norm = float(np.linalg.norm(result[0]))
        if abs(norm - 1.0) > 1e-5:
            print(f"  ✗ Not L2-normalized: norm={norm:.6f}")
            return False

        print(f"  ✓ Model loaded successfully")
        print(f"  ✓ Output shape: {result.shape}")
        print(f"  ✓ L2 norm: {norm:.6f}")
        print(f"  ✓ Model dimension: {engine.dimension}")
        return True

    except Exception as e:
        print(f"  ✗ Verification failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def print_summary() -> None:
    """Print model info summary."""
    print("\n  Model files after migration:")
    if MODEL_DIR.exists():
        total_size = 0
        for f in sorted(MODEL_DIR.iterdir()):
            if f.is_file():
                size_mb = f.stat().st_size / (1024 * 1024)
                total_size += f.stat().st_size
                print(f"    {f.name:30s} {size_mb:>8.1f} MB")
        print(f"    {'TOTAL':30s} {total_size / (1024 * 1024):>8.1f} MB")


def main() -> None:
    print("=" * 60)
    print(f"  Migrating to {NEW_REPO}")
    print("=" * 60)

    # Step 1: Backup
    print("\n[1/3] Backup current model...")
    backup_current()

    # Step 2: Download
    print(f"\n[2/3] Download {NEW_REPO}...")
    if not download_new_model():
        print("\n  ✗ Download failed — aborting.")
        print("  Existing backup is preserved at:", BACKUP_DIR)
        sys.exit(1)

    # Step 3: Verify
    print("\n[3/3] Verify model loads correctly...")
    if not verify_model():
        print("\n  ✗ Verification failed!")
        print("  To rollback:")
        print(f"    rm -rf {MODEL_DIR}")
        print(f"    mv {BACKUP_DIR} {MODEL_DIR}")
        sys.exit(1)

    # Summary
    print_summary()

    print("\n" + "=" * 60)
    print("  ✓ Migration complete!")
    print("=" * 60)
    print(f"  Model dir:  {MODEL_DIR}")
    print(f"  Backup dir: {BACKUP_DIR}")
    print()
    print("  NEXT STEPS (manual):")
    print("  1. Apply code changes in embeddings.py:")
    print("     - Update MODEL_NAME to 'intfloat/multilingual-e5-small'")
    print("     - Add QUERY_PREFIX / PASSAGE_PREFIX constants")
    print("     - Modify encode() to accept task='query'|'passage'")
    print("  2. Apply code change in server.py:")
    print("     - search_semantic: engine.encode([query], task='query')")
    print("  3. Re-embed all entities (new model + passage prefix):")
    print("     python scripts/reembed_all.py")
    print("  4. Restart MCP Memory server")


if __name__ == "__main__":
    main()
