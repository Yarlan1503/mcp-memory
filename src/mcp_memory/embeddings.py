"""Singleton ONNX embedding engine for semantic search."""

import struct
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path.home() / ".cache" / "mcp-memory-v2" / "models"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DIMENSION = 384


class EmbeddingEngine:
    """Singleton ONNX embedding engine.

    Loads ``model.onnx`` + ``tokenizer.json`` from a local directory
    (default ``~/.cache/mcp-memory-v2/models/``) and exposes an
    ``encode()`` method that returns L2-normalised float32 vectors
    suitable for cosine similarity search with sqlite-vec.
    """

    _instance: "EmbeddingEngine | None" = None

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "EmbeddingEngine":
        """Return the cached singleton.  ``available`` may be ``False``
        if the model files are missing on disk."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Drop the cached instance (useful for testing)."""
        cls._instance = None

    # ------------------------------------------------------------------
    # Init / load
    # ------------------------------------------------------------------

    def __init__(self, model_path: Path | None = None) -> None:
        self._available: bool = False
        self._session = None
        self._tokenizer = None
        self._input_names: list[str] = []

        load_dir = model_path or MODEL_DIR

        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            model_file = load_dir / "model.onnx"
            tokenizer_file = load_dir / "tokenizer.json"

            if not model_file.exists() or not tokenizer_file.exists():
                logger.warning(
                    "Model files not found at %s — embeddings unavailable",
                    load_dir,
                )
                return

            # ONNX session with all graph-level optimisations on CPU
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            self._session = ort.InferenceSession(
                str(model_file),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )

            # Discover expected input names from the ONNX graph
            self._input_names = [inp.name for inp in self._session.get_inputs()]
            logger.debug("ONNX model inputs: %s", self._input_names)

            # HuggingFace fast tokenizer
            self._tokenizer = Tokenizer.from_file(str(tokenizer_file))
            self._tokenizer.enable_truncation(max_length=512)
            self._tokenizer.enable_padding(length=512)

            self._available = True
            logger.info("Embedding engine loaded from %s", load_dir)

        except Exception as e:
            logger.error("Failed to load embedding model: %s", e)
            self._available = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """``True`` when the ONNX model and tokenizer loaded correctly."""
        return self._available

    @property
    def dimension(self) -> int:
        """Embedding dimensionality (384 for this model)."""
        return DIMENSION

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode *texts* into an L2-normalised ``(n, 384)`` float32 array.

        Pipeline: tokenise → ONNX forward → mean-pool → L2-normalise.

        Raises ``RuntimeError`` if the model is not loaded.
        """
        if not self._available:
            raise RuntimeError("Embedding model not available")

        # 1. Tokenise ---------------------------------------------------
        encoded = self._tokenizer.encode_batch(texts)

        input_ids = np.array(
            [e.ids for e in encoded],
            dtype=np.int64,
        )
        attention_mask = np.array(
            [e.attention_mask for e in encoded],
            dtype=np.int64,
        )

        # 2. Build ONNX feed dict (dynamic input discovery) -------------
        feed: dict[str, np.ndarray] = {}
        for name in self._input_names:
            if name == "input_ids":
                feed[name] = input_ids
            elif name == "attention_mask":
                feed[name] = attention_mask
            elif name == "token_type_ids":
                feed[name] = np.zeros_like(input_ids)
            else:
                logger.warning("Unknown ONNX input '%s' — filling with zeros", name)
                feed[name] = np.zeros_like(input_ids)

        # 3. ONNX inference ---------------------------------------------
        outputs = self._session.run(None, feed)
        token_embeddings = outputs[0]  # (batch, seq_len, 384)

        # 4. Mean pooling (mask out [PAD] tokens) -----------------------
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        mean_embeddings = sum_embeddings / sum_mask

        # 5. L2 normalise → float32 -------------------------------------
        norms = np.linalg.norm(mean_embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        normalized = mean_embeddings / norms

        return normalized.astype(np.float32)

    @staticmethod
    def prepare_entity_text(
        name: str,
        entity_type: str,
        observations: list[str],
    ) -> str:
        """Format an entity for embedding.

        Returns ``"{name} ({entity_type}): {obs1}. {obs2}. ..."``
        """
        obs_text = ". ".join(observations)
        return f"{name} ({entity_type}): {obs_text}"


# ------------------------------------------------------------------
# Binary helpers for sqlite-vec
# ------------------------------------------------------------------


def serialize_f32(vector: np.ndarray) -> bytes:
    """Pack a float32 vector into raw bytes for sqlite-vec.

    A 384-dim vector → 1 536 bytes.
    """
    return struct.pack(f"{len(vector)}f", *vector.flatten())


def deserialize_f32(data: bytes, dim: int = DIMENSION) -> np.ndarray:
    """Unpack raw bytes from sqlite-vec back into a float32 vector."""
    return np.frombuffer(data, dtype=np.float32).reshape(dim)
