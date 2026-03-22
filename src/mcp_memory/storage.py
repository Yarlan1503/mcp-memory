"""SQLite storage layer for the MCP Memory knowledge graph."""

import sqlite3

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MemoryStore:
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
        self.db.execute("PRAGMA busy_timeout=5000;")
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

    def close(self) -> None:
        """Close the database connection."""
        self.db.close()

    # ------------------------------------------------------------------
    # Schema initialisation
    # ------------------------------------------------------------------

    def init_db(self) -> None:
        """Execute CREATE TABLE / INDEX. Idempotent (IF NOT EXISTS)."""
        cur = self.db.cursor()

        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS entities (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL UNIQUE,
                entity_type TEXT    NOT NULL DEFAULT 'Generic',
                created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
                updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS observations (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id  INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                content    TEXT    NOT NULL,
                created_at TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS relations (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                from_entity   INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                to_entity     INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                relation_type TEXT    NOT NULL,
                created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
                UNIQUE(from_entity, to_entity, relation_type)
            );

            CREATE TABLE IF NOT EXISTS db_metadata (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )

        cur.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_obs_entity    ON observations(entity_id);
            CREATE INDEX IF NOT EXISTS idx_rel_from      ON relations(from_entity);
            CREATE INDEX IF NOT EXISTS idx_rel_to        ON relations(to_entity);
            CREATE INDEX IF NOT EXISTS idx_rel_type      ON relations(relation_type);
            CREATE INDEX IF NOT EXISTS idx_entities_name  ON entities(name);
            CREATE INDEX IF NOT EXISTS idx_entities_type  ON entities(entity_type);
            """
        )

        # sqlite-vec virtual table (best-effort)
        if self._vec_loaded:
            try:
                cur.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS entity_embeddings
                    USING vec0(embedding float[384] distance_metric=cosine);
                    """
                )
                logger.info("entity_embeddings virtual table created.")
            except Exception as exc:
                logger.error("Failed to create vec0 table: %s", exc)

        self.db.commit()

    # ------------------------------------------------------------------
    # Entity CRUD
    # ------------------------------------------------------------------

    def upsert_entity(self, name: str, entity_type: str) -> int:
        """INSERT or UPDATE entity. Returns entity_id. Updates updated_at."""
        self.db.execute(
            """
            INSERT INTO entities (name, entity_type)
            VALUES (?, ?)
            ON CONFLICT(name) DO UPDATE SET
                entity_type = excluded.entity_type,
                updated_at  = datetime('now');
            """,
            (name, entity_type),
        )
        self.db.commit()
        row = self.db.execute(
            "SELECT id FROM entities WHERE name = ?", (name,)
        ).fetchone()
        return row["id"]  # type: ignore[return-value]

    def get_entity_by_name(self, name: str) -> dict | None:
        """Returns entity dict or None."""
        row = self.db.execute(
            "SELECT id, name, entity_type, created_at, updated_at FROM entities WHERE name = ?",
            (name,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def get_entity_by_id(self, entity_id: int) -> dict | None:
        """Get entity by ID."""
        row = self.db.execute(
            "SELECT id, name, entity_type, created_at, updated_at FROM entities WHERE id = ?",
            (entity_id,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def get_all_entities(self) -> list[dict]:
        """All entities with their observations."""
        entities = [
            dict(r)
            for r in self.db.execute(
                "SELECT id, name, entity_type, created_at, updated_at FROM entities"
            ).fetchall()
        ]
        for entity in entities:
            entity["observations"] = self.get_observations(entity["id"])
        return entities

    def delete_entities_by_names(self, names: list[str]) -> int:
        """
        CASCADE delete. Returns count of deleted.
        CRITICAL: Delete embeddings first (vec0 doesn't support CASCADE),
        then delete entities. All in a single implicit transaction.
        """
        if not names:
            return 0

        placeholders = ",".join("?" for _ in names)
        ids = [
            r["id"]
            for r in self.db.execute(
                f"SELECT id FROM entities WHERE name IN ({placeholders})", names
            ).fetchall()
        ]

        if not ids:
            return 0

        id_placeholders = ",".join("?" for _ in ids)

        try:
            # 1. Delete embeddings (vec0 has no CASCADE support)
            if self._vec_loaded:
                try:
                    self.db.execute(
                        f"DELETE FROM entity_embeddings WHERE rowid IN ({id_placeholders})",
                        ids,
                    )
                except Exception:
                    logger.warning("Could not delete embeddings for entities %s", ids)

            # 2. Delete entities (CASCADE takes care of observations & relations)
            self.db.execute(
                f"DELETE FROM entities WHERE id IN ({id_placeholders})", ids
            )

            self.db.commit()
            return len(ids)
        except Exception:
            self.db.rollback()
            raise

    def search_entities(self, query: str) -> list[dict]:
        """
        LIKE search on name, entity_type, and observations.content.
        Returns entities with their observations.
        """
        pattern = f"%{query}%"
        rows = self.db.execute(
            """
            SELECT DISTINCT e.id, e.name, e.entity_type, e.created_at, e.updated_at
            FROM entities e
            LEFT JOIN observations o ON o.entity_id = e.id
            WHERE e.name LIKE ? OR e.entity_type LIKE ? OR o.content LIKE ?
            """,
            (pattern, pattern, pattern),
        ).fetchall()

        results: list[dict] = []
        for row in rows:
            entity = dict(row)
            entity["observations"] = self.get_observations(entity["id"])
            results.append(entity)
        return results

    # ------------------------------------------------------------------
    # Observation CRUD
    # ------------------------------------------------------------------

    def add_observations(self, entity_id: int, observations: list[str]) -> int:
        """INSERT multiple observations. Returns count inserted.
        Skips duplicates (same content for same entity)."""
        inserted = 0
        for content in observations:
            # Skip duplicates
            existing = self.db.execute(
                "SELECT 1 FROM observations WHERE entity_id = ? AND content = ?",
                (entity_id, content),
            ).fetchone()
            if existing:
                continue
            self.db.execute(
                "INSERT INTO observations (entity_id, content) VALUES (?, ?)",
                (entity_id, content),
            )
            inserted += 1
        if inserted:
            self.db.commit()
        return inserted

    def get_observations(self, entity_id: int) -> list[str]:
        """All observations for an entity (content only)."""
        rows = self.db.execute(
            "SELECT content FROM observations WHERE entity_id = ? ORDER BY id",
            (entity_id,),
        ).fetchall()
        return [r["content"] for r in rows]

    def delete_observations(self, entity_id: int, observations: list[str]) -> int:
        """DELETE by exact content match. Returns count deleted."""
        if not observations:
            return 0
        placeholders = ",".join("?" for _ in observations)
        cursor = self.db.execute(
            f"DELETE FROM observations WHERE entity_id = ? AND content IN ({placeholders})",
            [entity_id, *observations],
        )
        self.db.commit()
        return cursor.rowcount

    # ------------------------------------------------------------------
    # Relation CRUD
    # ------------------------------------------------------------------

    def create_relation(
        self, from_entity_id: int, to_entity_id: int, relation_type: str
    ) -> bool:
        """INSERT relation. ON CONFLICT -> return False (already exists)."""
        try:
            self.db.execute(
                """
                INSERT INTO relations (from_entity, to_entity, relation_type)
                VALUES (?, ?, ?);
                """,
                (from_entity_id, to_entity_id, relation_type),
            )
            self.db.commit()
            return True
        except sqlite3.IntegrityError:
            # UNIQUE constraint violated -> already exists
            return False

    def delete_relation(
        self, from_entity_id: int, to_entity_id: int, relation_type: str
    ) -> bool:
        """DELETE relation. Returns True if it existed."""
        cursor = self.db.execute(
            """
            DELETE FROM relations
            WHERE from_entity = ? AND to_entity = ? AND relation_type = ?;
            """,
            (from_entity_id, to_entity_id, relation_type),
        )
        self.db.commit()
        return cursor.rowcount > 0

    def get_all_relations(self) -> list[dict]:
        """All relations with entity names (JOIN).
        Returns list of {"from": entity_name, "to": entity_name, "relationType": relation_type}."""
        rows = self.db.execute(
            """
            SELECT e1.name AS from_name, e2.name AS to_name, r.relation_type
            FROM relations r
            JOIN entities e1 ON r.from_entity = e1.id
            JOIN entities e2 ON r.to_entity   = e2.id;
            """
        ).fetchall()
        return [
            {
                "from": r["from_name"],
                "to": r["to_name"],
                "relationType": r["relation_type"],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Embedding ops
    # ------------------------------------------------------------------

    def store_embedding(self, entity_id: int, embedding: bytes) -> None:
        """INSERT OR REPLACE embedding. entity_id is rowid in vec0."""
        if not self._vec_loaded:
            logger.warning("sqlite-vec not loaded — cannot store embedding.")
            return
        try:
            self.db.execute(
                "INSERT OR REPLACE INTO entity_embeddings(rowid, embedding) VALUES (?, ?)",
                (entity_id, embedding),
            )
            self.db.commit()
        except Exception as exc:
            self.db.rollback()
            logger.error("Failed to store embedding for entity %s: %s", entity_id, exc)

    def delete_embedding(self, entity_id: int) -> None:
        """DELETE embedding by rowid."""
        if not self._vec_loaded:
            return
        try:
            self.db.execute(
                "DELETE FROM entity_embeddings WHERE rowid = ?",
                (entity_id,),
            )
            self.db.commit()
        except Exception as exc:
            self.db.rollback()
            logger.error("Failed to delete embedding for entity %s: %s", entity_id, exc)

    def search_embeddings(self, query_embedding: bytes, limit: int = 10) -> list[dict]:
        """KNN search. Returns list of {"entity_id": int, "distance": float}."""
        if not self._vec_loaded:
            logger.warning("sqlite-vec not loaded — cannot search embeddings.")
            return []
        try:
            rows = self.db.execute(
                """
                SELECT rowid, distance
                FROM entity_embeddings
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT ?;
                """,
                (query_embedding, limit),
            ).fetchall()
            return [
                {"entity_id": r["rowid"], "distance": float(r["distance"])}
                for r in rows
            ]
        except Exception as exc:
            logger.error("Embedding search failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_metadata(self, key: str) -> str | None:
        """Get metadata value."""
        row = self.db.execute(
            "SELECT value FROM db_metadata WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def set_metadata(self, key: str, value: str) -> None:
        """Set metadata value (INSERT OR REPLACE)."""
        self.db.execute(
            "INSERT OR REPLACE INTO db_metadata (key, value) VALUES (?, ?)",
            (key, value),
        )
        self.db.commit()

    # ------------------------------------------------------------------
    # Read graph (Anthropic-compatible format)
    # ------------------------------------------------------------------

    def read_graph(self) -> dict:
        """Returns entire graph in Anthropic-compatible format:
        {
            "entities": [{"name": "...", "entityType": "...", "observations": [...]}],
            "relations": [{"from": "...", "to": "...", "relationType": "..."}]
        }
        """
        entities_rows = self.db.execute(
            "SELECT id, name, entity_type FROM entities"
        ).fetchall()

        entities: list[dict] = []
        for row in entities_rows:
            obs = self.get_observations(row["id"])
            entities.append(
                {
                    "name": row["name"],
                    "entityType": row["entity_type"],
                    "observations": obs,
                }
            )

        relations = self.get_all_relations()

        return {"entities": entities, "relations": relations}
