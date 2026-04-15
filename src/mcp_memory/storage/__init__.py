"""MCP Memory storage layer — modular MemoryStore backed by SQLite."""

import logging
import sqlite3
from pathlib import Path

from mcp_memory.retry import retry_on_locked
from mcp_memory.storage._constants import (
    INVERSE_RELATIONS,
    LEGACY_RELATION_TYPES,
    VALID_AUTHORS,
    VALID_MOODS,
    VALID_TARGET_TYPES,
)
from mcp_memory.storage.access import AccessMixin
from mcp_memory.storage.consolidation import ConsolidationMixin
from mcp_memory.storage.core import CoreMixin
from mcp_memory.storage.reflections import ReflectionsMixin
from mcp_memory.storage.relations import RelationsMixin
from mcp_memory.storage.schema import SchemaMixin
from mcp_memory.storage.search import SearchMixin

logger = logging.getLogger(__name__)

__all__ = [
    "MemoryStore",
    "INVERSE_RELATIONS",
    "LEGACY_RELATION_TYPES",
    "VALID_TARGET_TYPES",
    "VALID_AUTHORS",
    "VALID_MOODS",
]


class MemoryStore(
    SchemaMixin,
    CoreMixin,
    RelationsMixin,
    SearchMixin,
    AccessMixin,
    ReflectionsMixin,
    ConsolidationMixin,
):
    """Persistent memory store backed by SQLite with optional sqlite-vec embeddings."""

    def __init__(
        self, db_path: str = "~/.config/opencode/mcp-memory/memory.db"
    ) -> None:
        """
        Opens SQLite connection with WAL mode.
        Resolves path with pathlib.Path.expanduser().
        Creates parent directory if needed.
        Uses check_same_thread=False.
        """
        if db_path == ":memory:":
            self.db = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            resolved = Path(db_path).expanduser().resolve()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(str(resolved), check_same_thread=False)
        self.db.row_factory = sqlite3.Row

        # --- PRAGMAs (before any table creation) ---
        self.db.execute("PRAGMA journal_mode=WAL;")
        self.db.execute("PRAGMA busy_timeout=10000;")
        self.db.execute("PRAGMA synchronous=NORMAL;")
        self.db.execute("PRAGMA cache_size=-64000;")
        self.db.execute("PRAGMA temp_store=MEMORY;")
        self.db.execute("PRAGMA foreign_keys=ON;")

        # --- Load sqlite-vec extension (optional) ---
        self._vec_loaded: bool = False
        try:
            self.db.enable_load_extension(True)
            import sqlite_vec  # noqa: F401 — ensure importable

            sqlite_vec.load(self.db)
            self.db.enable_load_extension(False)
            self._vec_loaded = True
            logger.info("sqlite-vec extension loaded successfully.")
        except Exception:
            self.db.enable_load_extension(False)
            logger.warning(
                "sqlite-vec extension could not be loaded. "
                "Embedding operations will be unavailable."
            )

        # --- Check FTS5 availability ---
        self._fts_available: bool = False
        try:
            self.db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS _fts_test USING fts5(dummy);"
            )
            self.db.execute("DROP TABLE IF EXISTS _fts_test;")
            self._fts_available = True
            logger.info("FTS5 extension available.")
        except Exception:
            logger.warning("FTS5 not available. Full-text search will be disabled.")

    def close(self) -> None:
        """Close the database connection."""
        self.db.close()

    def _get_embedding_engine(self):
        """Get embedding engine instance (best-effort). Returns None if unavailable."""
        try:
            from mcp_memory.embeddings import EmbeddingEngine

            engine = EmbeddingEngine.get_instance()
            if engine and engine.available:
                return engine
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Metadata (small enough to live in the facade)
    # ------------------------------------------------------------------

    def get_metadata(self, key: str) -> str | None:
        """Get metadata value."""
        row = self.db.execute(
            "SELECT value FROM db_metadata WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    @retry_on_locked
    def set_metadata(self, key: str, value: str) -> None:
        """Set metadata value (INSERT OR REPLACE)."""
        self.db.execute(
            "INSERT OR REPLACE INTO db_metadata (key, value) VALUES (?, ?)",
            (key, value),
        )
        self.db.commit()
