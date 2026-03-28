"""Singleton ONNX embedding engine for semantic search."""

import re
import struct
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path.home() / ".cache" / "mcp-memory-v2" / "models"
MODEL_NAME = "intfloat/multilingual-e5-small"
DIMENSION = 384

# E5-small requires task prefixes for asymmetric retrieval
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

# Token budget constants for prepare_entity_text
OVERHEAD_TOKENS = 25  # Tokens para "name (type): " + "Rel: ..." overhead
MAX_TOKENS = 480  # Budget total (512 - buffer)
CHARS_PER_TOKEN = 3.57  # Ratio empírico para español (~123 chars / 35 tokens)


def _estimate_tokens(text: str) -> int:
    """Estimate token count using empirical chars/token ratio for Spanish."""
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def _lexical_diversity(text: str) -> float:
    """Compute lexical diversity: unique words / total words.
    Higher = more diverse vocabulary."""
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


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

    def encode(self, texts: list[str], *, task: str = "passage") -> np.ndarray:
        """Encode *texts* into an L2-normalised ``(n, 384)`` float32 array.

        Pipeline: prefix → tokenise → ONNX forward → mean-pool → L2-normalise.

        The ``task`` parameter controls the E5-small prefix:
        ``"query"`` for search queries, ``"passage"`` (default) for stored text.

        Raises ``RuntimeError`` if the model is not loaded.
        """
        if not texts:
            return np.empty((0, DIMENSION))
        if not self._available:
            raise RuntimeError("Embedding model not available")

        # 0. Apply e5-small task prefix ----------------------------------
        prefix = QUERY_PREFIX if task == "query" else PASSAGE_PREFIX
        prefixed = [prefix + t for t in texts]

        # 1. Tokenise ---------------------------------------------------
        encoded = self._tokenizer.encode_batch(prefixed)

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
        relations: list[dict] | None = None,
    ) -> str:
        """Format an entity for embedding with Head+Tail+Diversity selection.

        Strategy:
        1. Head (first 3 obs): definitional, structural info
        2. Tail (last 7 obs): most recent state
        3. Middle: fill budget with most lexically diverse observations
        4. Relations: append as context ("Rel: type -> target; ...")

        Uses ` | ` separator between observations for semantic clarity.
        Budget: MAX_TOKENS (480) estimated via CHARS_PER_TOKEN ratio.
        """
        if not observations:
            base = f"{name} ({entity_type})"
        else:
            # Build relation context string
            rel_parts: list[str] = []
            rel_token_budget = 0
            if relations:
                for rel in relations:
                    target = rel.get(
                        "target_name", f"id:{rel.get('to_id', rel.get('from_id', '?'))}"
                    )
                    rel_type = rel.get("relation_type", "related_to")
                    rel_parts.append(f"{rel_type} → {target}")

            rel_text = ""
            if rel_parts:
                rel_text = " | Rel: " + "; ".join(rel_parts)
                rel_token_budget = _estimate_tokens(rel_text)

            # Available budget for observations
            header = f"{name} ({entity_type}): "
            header_tokens = _estimate_tokens(header)
            obs_budget = MAX_TOKENS - OVERHEAD_TOKENS - header_tokens - rel_token_budget
            obs_char_budget = int(obs_budget * CHARS_PER_TOKEN)

            # --- Selection strategy ---
            n = len(observations)

            # Always take head (first 3)
            head = observations[:3]

            if n <= 3:
                # Very few observations — use all
                selected = list(observations)
            elif n <= 10:
                # Moderate — take head + tail (all remaining from end)
                tail_count = min(7, n - 3)
                tail = observations[-tail_count:]
                # Middle = everything between head and tail
                middle_start = 3
                middle_end = n - tail_count
                middle = observations[middle_start:middle_end]
                selected = head + middle + tail
            else:
                # Many observations — Head + Tail + Diversity middle
                tail = observations[-7:]
                # Middle candidates = everything between head and tail
                middle_candidates = observations[3:-7]

                # Build output with head first
                selected = list(head)
                used_chars = sum(len(o) for o in selected) + 3 * 3  # 3 separators " | "

                # Add tail (reserve space)
                tail_chars = sum(len(o) for o in tail) + 7 * 3  # 7 separators
                remaining_chars = obs_char_budget - used_chars - tail_chars

                # Sort middle candidates by lexical diversity (descending)
                middle_scored = sorted(
                    middle_candidates,
                    key=_lexical_diversity,
                    reverse=True,
                )

                # Fill remaining budget with most diverse middle observations
                diverse_middle: list[str] = []
                for obs in middle_scored:
                    obs_cost = len(obs) + 3  # +3 for separator
                    if remaining_chars >= obs_cost:
                        diverse_middle.append(obs)
                        remaining_chars -= obs_cost
                    if remaining_chars < 20:  # minimum useful observation
                        break

                # Re-order diverse_middle by original position for coherence
                middle_set = set(id(o) for o in diverse_middle)
                ordered_middle = [o for o in middle_candidates if id(o) in middle_set]

                selected = head + ordered_middle + tail

            # Join with semantic separator
            obs_text = " | ".join(selected)
            base = f"{header}{obs_text}{rel_text}"

        return base


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
