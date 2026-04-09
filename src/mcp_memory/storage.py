"""SQLite storage layer for the MCP Memory knowledge graph."""

import sqlite3

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# --- Phase 3: Inverse relation map ---
INVERSE_RELATIONS: dict[str, str] = {
    "contiene": "parte_de",
    "parte_de": "contiene",
}

# --- Phase 3: Legacy relation type normalization ---
LEGACY_RELATION_TYPES: dict[str, tuple[str, str]] = {
    "continua": ("contribuye_a", "sesión continuación"),
    "documentado_en": ("producido_por", "documentado en"),
}

# --- Phase 4: Reflections validation constants ---
VALID_TARGET_TYPES = {"entity", "session", "relation", "global"}
VALID_AUTHORS = {"nolan", "sofia"}
VALID_MOODS = {"frustracion", "satisfaccion", "curiosidad", "duda", "insight"}


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

    # ------------------------------------------------------------------
    # Embedding engine helper
    # ------------------------------------------------------------------

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

            CREATE TABLE IF NOT EXISTS entity_access (
                entity_id    INTEGER PRIMARY KEY REFERENCES entities(id) ON DELETE CASCADE,
                access_count INTEGER NOT NULL DEFAULT 1,
                last_access  TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS co_occurrences (
                entity_a_id  INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                entity_b_id  INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                co_count     INTEGER NOT NULL DEFAULT 1,
                last_co      TEXT    NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (entity_a_id, entity_b_id)
            );

            CREATE TABLE IF NOT EXISTS search_events (
                event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text   TEXT    NOT NULL,
                query_hash   TEXT    NOT NULL,
                timestamp    TEXT    NOT NULL DEFAULT (datetime('now')),
                treatment    INTEGER NOT NULL,
                k_limit      INTEGER NOT NULL,
                num_results  INTEGER NOT NULL,
                duration_ms  REAL,
                engine_used  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS search_results (
                result_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id       INTEGER NOT NULL REFERENCES search_events(event_id),
                entity_id      INTEGER NOT NULL,
                entity_name    TEXT    NOT NULL,
                rank           INTEGER NOT NULL,
                limbic_score   REAL,
                cosine_sim     REAL,
                importance     REAL,
                temporal       REAL,
                cooc_boost     REAL,
                baseline_rank  INTEGER,
                UNIQUE(event_id, entity_id)
            );

            CREATE TABLE IF NOT EXISTS implicit_feedback (
                feedback_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id      INTEGER NOT NULL REFERENCES search_events(event_id),
                entity_id     INTEGER NOT NULL,
                re_accessed   INTEGER NOT NULL DEFAULT 0,
                access_delta  INTEGER,
                session_id    TEXT,
                FOREIGN KEY (event_id, entity_id) REFERENCES search_results(event_id, entity_id)
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
            CREATE INDEX IF NOT EXISTS idx_access_last    ON entity_access(last_access);
            CREATE INDEX IF NOT EXISTS idx_cooc_b         ON co_occurrences(entity_b_id);
            CREATE INDEX IF NOT EXISTS idx_search_events_hash  ON search_events(query_hash);
            CREATE INDEX IF NOT EXISTS idx_search_events_time  ON search_events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_search_results_event ON search_results(event_id);
            CREATE INDEX IF NOT EXISTS idx_search_results_entity ON search_results(entity_id);
            CREATE INDEX IF NOT EXISTS idx_implicit_event      ON implicit_feedback(event_id);
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

        # FTS5 full-text search table (best-effort)
        if self._fts_available:
            try:
                cur.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS entity_fts
                    USING fts5(name, entity_type, obs_text, tokenize="unicode61");
                    """
                )
                logger.info("entity_fts virtual table created.")

                # Backfill from existing entities if FTS table is empty
                fts_count = cur.execute("SELECT COUNT(*) FROM entity_fts").fetchone()[0]
                entity_count = cur.execute("SELECT COUNT(*) FROM entities").fetchone()[
                    0
                ]
                if fts_count == 0 and entity_count > 0:
                    self._backfill_fts()
            except Exception as exc:
                logger.error("Failed to create FTS5 table: %s", exc)

        self.db.commit()

        # --- Idempotent migrations ---
        self._add_similarity_flag_column()
        self._add_entity_access_log_table()
        self._add_kind_column()
        self._add_supersedes_columns()
        self._migrate_status_column()
        self._migrate_relation_fields()
        self._migrate_relation_types()

        # --- Phase 4: Reflections table ---
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS reflections (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                target_type TEXT NOT NULL,
                target_id   INTEGER,
                author      TEXT NOT NULL,
                content     TEXT NOT NULL,
                mood        TEXT,
                created_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );
            """
        )

        # Phase 4: FTS5 index for reflections (content sync manual)
        if self._fts_available:
            try:
                cur.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS reflection_fts USING fts5(
                        content,
                        author,
                        mood,
                        content='reflections',
                        content_rowid='id'
                    );
                    """
                )
                logger.info("reflection_fts virtual table created.")
            except Exception as exc:
                logger.error("Failed to create reflection_fts table: %s", exc)

        # Phase 4: Embedding index for reflections
        if self._vec_loaded:
            try:
                cur.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS reflection_embeddings
                    USING vec0(embedding float[384]);
                    """
                )
                logger.info("reflection_embeddings virtual table created.")
            except Exception as exc:
                logger.error("Failed to create reflection_embeddings table: %s", exc)

        self.db.commit()

    def _add_entity_access_log_table(self) -> None:
        """Idempotent migration: create entity_access_log table for day-level access tracking."""
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS entity_access_log (
                entity_id    INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                access_date  TEXT    NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (entity_id, access_date)
            );
            """
        )
        self.db.commit()
        logger.info("entity_access_log table verified.")

    def _add_similarity_flag_column(self) -> None:
        """Idempotent migration: add similarity_flag column to observations if missing."""
        try:
            cols = self.db.execute("PRAGMA table_info(observations)").fetchall()
            col_names = {c["name"] for c in cols}
            if "similarity_flag" not in col_names:
                self.db.execute(
                    "ALTER TABLE observations ADD COLUMN similarity_flag "
                    "INTEGER NOT NULL DEFAULT 0"
                )
                self.db.commit()
                logger.info("Added similarity_flag column to observations table.")
        except Exception as exc:
            logger.warning("Failed to add similarity_flag column: %s", exc)

    def _add_kind_column(self) -> None:
        """Idempotent migration: add kind column to observations if missing."""
        try:
            cols = self.db.execute("PRAGMA table_info(observations)").fetchall()
            col_names = {c["name"] for c in cols}
            if "kind" not in col_names:
                self.db.execute(
                    "ALTER TABLE observations ADD COLUMN kind "
                    "TEXT NOT NULL DEFAULT 'generic'"
                )
                self.db.commit()
                logger.info("Added kind column to observations table.")
        except Exception as exc:
            logger.warning("Failed to add kind column: %s", exc)

    def _add_supersedes_columns(self) -> None:
        """Idempotent migration: add supersedes and superseded_at columns to observations if missing."""
        try:
            cols = self.db.execute("PRAGMA table_info(observations)").fetchall()
            col_names = {c["name"] for c in cols}
            if "supersedes" not in col_names:
                self.db.execute(
                    "ALTER TABLE observations ADD COLUMN supersedes "
                    "INTEGER NULL REFERENCES observations(id)"
                )
                self.db.commit()
                logger.info("Added supersedes column to observations table.")
            if "superseded_at" not in col_names:
                self.db.execute(
                    "ALTER TABLE observations ADD COLUMN superseded_at TEXT NULL"
                )
                self.db.commit()
                logger.info("Added superseded_at column to observations table.")
        except Exception as exc:
            logger.warning("Failed to add supersedes columns: %s", exc)

    def _migrate_status_column(self) -> None:
        """Idempotent migration: add status column to entities if missing."""
        try:
            cols = self.db.execute("PRAGMA table_info(entities)").fetchall()
            col_names = {c["name"] for c in cols}
            if "status" not in col_names:
                self.db.execute(
                    "ALTER TABLE entities ADD COLUMN status "
                    "TEXT NOT NULL DEFAULT 'activo'"
                )
                self.db.commit()
                logger.info("Added status column to entities table.")
        except Exception as exc:
            logger.warning("Failed to add status column: %s", exc)

    def _migrate_relation_fields(self) -> None:
        """Idempotent migration: add context, active, ended_at columns to relations if missing."""
        try:
            cols = self.db.execute("PRAGMA table_info(relations)").fetchall()
            col_names = {c["name"] for c in cols}
            if "context" not in col_names:
                self.db.execute("ALTER TABLE relations ADD COLUMN context TEXT NULL")
                self.db.commit()
                logger.info("Added context column to relations table.")
            if "active" not in col_names:
                self.db.execute(
                    "ALTER TABLE relations ADD COLUMN active INTEGER NOT NULL DEFAULT 1"
                )
                self.db.commit()
                logger.info("Added active column to relations table.")
            if "ended_at" not in col_names:
                self.db.execute("ALTER TABLE relations ADD COLUMN ended_at TEXT NULL")
                self.db.commit()
                logger.info("Added ended_at column to relations table.")
        except Exception as exc:
            logger.warning("Failed to add relation fields: %s", exc)

    def _migrate_relation_types(self) -> None:
        """Idempotent migration: normalize legacy relation types.
        Updates legacy types to their modern equivalents, setting context
        only if the existing context is NULL."""
        try:
            for legacy_type, (new_type, reason) in LEGACY_RELATION_TYPES.items():
                cursor = self.db.execute(
                    "UPDATE relations SET relation_type = ?, "
                    "context = CASE WHEN context IS NULL THEN ? ELSE context END "
                    "WHERE relation_type = ?",
                    (new_type, reason, legacy_type),
                )
                if cursor.rowcount > 0:
                    self.db.commit()
                    logger.info(
                        "Migrated %d relations from '%s' to '%s'.",
                        cursor.rowcount,
                        legacy_type,
                        new_type,
                    )
        except Exception as exc:
            logger.warning("Failed to migrate relation types: %s", exc)

    # ------------------------------------------------------------------
    # Entity CRUD
    # ------------------------------------------------------------------

    def upsert_entity(self, name: str, entity_type: str, status: str = "activo") -> int:
        """INSERT or UPDATE entity. Returns entity_id. Updates updated_at."""
        self.db.execute(
            """
            INSERT INTO entities (name, entity_type, status)
            VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                entity_type = excluded.entity_type,
                status = excluded.status,
                updated_at  = datetime('now');
            """,
            (name, entity_type, status),
        )
        self.db.commit()
        row = self.db.execute(
            "SELECT id FROM entities WHERE name = ?", (name,)
        ).fetchone()
        entity_id = row["id"]  # type: ignore[return-value]
        self._sync_fts(entity_id)
        return entity_id

    def get_entity_by_name(self, name: str) -> dict | None:
        """Returns entity dict or None."""
        row = self.db.execute(
            "SELECT id, name, entity_type, status, created_at, updated_at FROM entities WHERE name = ?",
            (name,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def get_entity_by_id(self, entity_id: int) -> dict | None:
        """Get entity by ID."""
        row = self.db.execute(
            "SELECT id, name, entity_type, status, created_at, updated_at FROM entities WHERE id = ?",
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
                "SELECT id, name, entity_type, status, created_at, updated_at FROM entities"
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

        with self.db:
            try:
                # 1. Delete embeddings (vec0 has no CASCADE support)
                if self._vec_loaded:
                    try:
                        self.db.execute(
                            f"DELETE FROM entity_embeddings WHERE rowid IN ({id_placeholders})",
                            ids,
                        )
                    except Exception:
                        logger.warning(
                            "Could not delete embeddings for entities %s", ids
                        )

                # 2. Delete FTS entries (FTS5 doesn't CASCADE either)
                if self._fts_available:
                    try:
                        self.db.execute(
                            f"DELETE FROM entity_fts WHERE rowid IN ({id_placeholders})",
                            ids,
                        )
                    except Exception:
                        logger.warning(
                            "Could not delete FTS entries for entities %s", ids
                        )

                # 3. Delete entities (CASCADE takes care of observations & relations)
                self.db.execute(
                    f"DELETE FROM entities WHERE id IN ({id_placeholders})", ids
                )

                return len(ids)
            except Exception:
                raise

    def search_entities(self, query: str) -> list[dict]:
        """
        LIKE search on name, entity_type, and observations.content.
        Returns entities with their observations.
        """
        pattern = f"%{query}%"
        rows = self.db.execute(
            """
            SELECT DISTINCT e.id, e.name, e.entity_type, e.status, e.created_at, e.updated_at
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

    def add_observations(
        self,
        entity_id: int,
        observations: list[str],
        kind: str = "generic",
        supersedes: int | None = None,
    ) -> int:
        """INSERT multiple observations. Returns count inserted.
        Skips duplicates (same content for same entity).
        Observations semantically similar to existing ones (cosine >= 0.85)
        are inserted with similarity_flag=1 for later review.

        Args:
            entity_id: The entity to add observations to.
            observations: List of observation strings.
            kind: The kind/type of observations (default 'generic').
            supersedes: Optional observation ID to supersede. The referenced obs
                will be marked with superseded_at, and the new obs will reference it.
        """
        if not observations:
            return 0

        # --- Handle supersedes logic ---
        if supersedes is not None:
            # Verify the observation exists and belongs to this entity
            row = self.db.execute(
                "SELECT id, entity_id, superseded_at FROM observations WHERE id = ?",
                (supersedes,),
            ).fetchone()
            if row is None or row["entity_id"] != entity_id:
                logger.warning(
                    "Supersedes ID %s does not exist or belongs to different entity (expected %s). Ignoring.",
                    supersedes,
                    entity_id,
                )
                supersedes = None
            elif row["superseded_at"] is not None:
                logger.warning(
                    "Observation %s is already superseded (at %s). Ignoring supersedes.",
                    supersedes,
                    row["superseded_at"],
                )
                supersedes = None

        inserted = 0

        # Try to get embedding engine for semantic dedup
        engine = self._get_embedding_engine()
        existing_obs = self.get_observations(entity_id)

        if existing_obs and engine is not None:
            # --- Semantic dedup path ---
            try:
                import numpy as np  # noqa: F401

                # Batch encode all existing observations
                existing_embeddings = engine.encode(existing_obs)

                for content in observations:
                    # Skip exact duplicates
                    exact = self.db.execute(
                        "SELECT 1 FROM observations WHERE entity_id = ? AND content = ?",
                        (entity_id, content),
                    ).fetchone()
                    if exact:
                        continue

                    # Encode the new observation
                    new_embedding = engine.encode([content])  # (1, 384)

                    # Cosine similarity with all existing (L2-normalised → dot product)
                    similarities = existing_embeddings @ new_embedding[0]  # (n,)
                    max_sim = (
                        float(np.max(similarities)) if len(similarities) > 0 else 0.0
                    )

                    # Find the index of the most similar existing observation
                    max_idx = (
                        int(np.argmax(similarities)) if len(similarities) > 0 else 0
                    )
                    best_existing = existing_obs[max_idx] if existing_obs else ""

                    # Combined similarity: cosine OR containment for asymmetric pairs
                    from mcp_memory.scoring import combined_similarity

                    flag = (
                        1 if combined_similarity(max_sim, content, best_existing) else 0
                    )

                    self.db.execute(
                        "INSERT INTO observations (entity_id, content, similarity_flag, kind) "
                        "VALUES (?, ?, ?, ?)",
                        (entity_id, content, flag, kind),
                    )
                    inserted += 1

            except Exception as exc:
                logger.warning(
                    "Semantic dedup failed, falling back to normal insert: %s", exc
                )
                # Fallback: insert remaining without flag
                for content in observations:
                    exact = self.db.execute(
                        "SELECT 1 FROM observations WHERE entity_id = ? AND content = ?",
                        (entity_id, content),
                    ).fetchone()
                    if exact:
                        continue
                    self.db.execute(
                        "INSERT INTO observations (entity_id, content, similarity_flag, kind) "
                        "VALUES (?, ?, 0, ?)",
                        (entity_id, content, kind),
                    )
                    inserted += 1
        else:
            # --- Normal path: no engine or no existing observations ---
            for content in observations:
                exact = self.db.execute(
                    "SELECT 1 FROM observations WHERE entity_id = ? AND content = ?",
                    (entity_id, content),
                ).fetchone()
                if exact:
                    continue
                self.db.execute(
                    "INSERT INTO observations (entity_id, content, similarity_flag, kind) "
                    "VALUES (?, ?, 0, ?)",
                    (entity_id, content, kind),
                )
                inserted += 1

        # --- Apply supersedes after insertion ---
        if supersedes is not None and inserted > 0:
            # Mark old observation as superseded
            self.db.execute(
                "UPDATE observations SET superseded_at = datetime('now') WHERE id = ?",
                (supersedes,),
            )
            # Get the ID of the last inserted observation for this entity
            new_obs_row = self.db.execute(
                "SELECT id FROM observations WHERE entity_id = ? ORDER BY id DESC LIMIT 1",
                (entity_id,),
            ).fetchone()
            if new_obs_row:
                self.db.execute(
                    "UPDATE observations SET supersedes = ? WHERE id = ?",
                    (supersedes, new_obs_row["id"]),
                )

        if inserted:
            self.db.commit()
            self._sync_fts(entity_id)
        return inserted

    def get_observations(
        self, entity_id: int, exclude_superseded: bool = True
    ) -> list[str]:
        """All observations for an entity (content only).

        Args:
            entity_id: The entity ID.
            exclude_superseded: If True, exclude observations that have been superseded.
        """
        if exclude_superseded:
            rows = self.db.execute(
                "SELECT content FROM observations WHERE entity_id = ? AND superseded_at IS NULL ORDER BY id",
                (entity_id,),
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT content FROM observations WHERE entity_id = ? ORDER BY id",
                (entity_id,),
            ).fetchall()
        return [r["content"] for r in rows]

    def get_observations_with_ids(
        self, entity_id: int, exclude_superseded: bool = True
    ) -> list[dict]:
        """All observations for an entity with id, content, similarity_flag, kind, supersedes, superseded_at.

        Args:
            entity_id: The entity ID.
            exclude_superseded: If True, exclude observations that have been superseded.
        """
        if exclude_superseded:
            rows = self.db.execute(
                "SELECT id, content, similarity_flag, kind, supersedes, superseded_at "
                "FROM observations WHERE entity_id = ? AND superseded_at IS NULL ORDER BY id",
                (entity_id,),
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT id, content, similarity_flag, kind, supersedes, superseded_at "
                "FROM observations WHERE entity_id = ? ORDER BY id",
                (entity_id,),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "content": r["content"],
                "similarity_flag": r["similarity_flag"],
                "kind": r["kind"] if r["kind"] is not None else "generic",
                "supersedes": r["supersedes"],
                "superseded_at": r["superseded_at"],
            }
            for r in rows
        ]

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
        self._sync_fts(entity_id)
        return cursor.rowcount

    # ------------------------------------------------------------------
    # Relation CRUD
    # ------------------------------------------------------------------

    def create_relation(
        self,
        from_entity_id: int,
        to_entity_id: int,
        relation_type: str,
        context: str | None = None,
    ) -> bool:
        """INSERT relation. ON CONFLICT -> return False (already exists).
        If relation_type has an inverse, automatically creates the inverse relation."""
        try:
            self.db.execute(
                """
                INSERT INTO relations (from_entity, to_entity, relation_type, context)
                VALUES (?, ?, ?, ?);
                """,
                (from_entity_id, to_entity_id, relation_type, context),
            )
            self.db.commit()
            # Auto-create inverse relation (best-effort)
            self._ensure_inverse_relation(
                from_entity_id, to_entity_id, relation_type, context
            )
            return True
        except sqlite3.IntegrityError:
            # UNIQUE constraint violated -> already exists
            return False

    def _ensure_inverse_relation(
        self,
        from_id: int,
        to_id: int,
        relation_type: str,
        context: str | None = None,
    ) -> None:
        """Auto-create the inverse relation if one is defined in INVERSE_RELATIONS.
        Best-effort: silently skips if inverse already exists or type has no inverse."""
        inverse_type = INVERSE_RELATIONS.get(relation_type)
        if inverse_type is None:
            return

        # Build context for the inverse
        inverse_context = f"Inversa automática de {relation_type}"

        try:
            # Check if inverse already exists to avoid IntegrityError noise
            existing = self.db.execute(
                "SELECT 1 FROM relations WHERE from_entity = ? AND to_entity = ? AND relation_type = ?",
                (to_id, from_id, inverse_type),
            ).fetchone()
            if existing:
                return

            self.db.execute(
                "INSERT INTO relations (from_entity, to_entity, relation_type, context) VALUES (?, ?, ?, ?)",
                (to_id, from_id, inverse_type, inverse_context),
            )
            self.db.commit()
        except sqlite3.IntegrityError:
            # Race condition or already exists — ignore
            pass

    def _end_relation(self, relation_id: int) -> bool:
        """Deactivate a relation by setting active=0 and ended_at.
        Returns True if the relation was found and updated, False otherwise."""
        cursor = self.db.execute(
            "UPDATE relations SET active = 0, ended_at = datetime('now') WHERE id = ?",
            (relation_id,),
        )
        self.db.commit()
        return cursor.rowcount > 0

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
        Returns list of {"from": entity_name, "to": entity_name, "relationType": relation_type,
                          "context": str|None, "active": 1|0, "ended_at": str|None}."""
        rows = self.db.execute(
            """
            SELECT e1.name AS from_name, e2.name AS to_name, r.relation_type,
                   r.context, r.active, r.ended_at
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
                "context": r["context"],
                "active": r["active"],
                "ended_at": r["ended_at"],
            }
            for r in rows
        ]

    def get_relations_for_entity(self, entity_id: int) -> list[dict]:
        """Get all relations for an entity (both from and to), with resolved entity names.

        Returns list of dicts with relation_type, target_name, direction,
        context, active, and ended_at.
        """
        rows = self.db.execute(
            """
            SELECT r.id, r.from_entity, r.to_entity, r.relation_type,
                   r.context, r.active, r.ended_at,
                   e_from.name AS from_name, e_to.name AS to_name
            FROM relations r
            JOIN entities e_from ON r.from_entity = e_from.id
            JOIN entities e_to ON r.to_entity = e_to.id
            WHERE r.from_entity = ? OR r.to_entity = ?
            """,
            (entity_id, entity_id),
        ).fetchall()

        result = []
        for r in rows:
            if r["from_entity"] == entity_id:
                result.append(
                    {
                        "relation_type": r["relation_type"],
                        "target_name": r["to_name"],
                        "direction": "from",
                        "context": r["context"],
                        "active": r["active"],
                        "ended_at": r["ended_at"],
                    }
                )
            else:
                result.append(
                    {
                        "relation_type": r["relation_type"],
                        "target_name": r["from_name"],
                        "direction": "to",
                        "context": r["context"],
                        "active": r["active"],
                        "ended_at": r["ended_at"],
                    }
                )
        return result

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
    # Limbic: Access tracking
    # ------------------------------------------------------------------

    def init_access(self, entity_id: int) -> None:
        """Initialize access tracking for a new entity (access_count=1)."""
        self.db.execute(
            "INSERT OR IGNORE INTO entity_access (entity_id, access_count, last_access) VALUES (?, 1, datetime('now'))",
            (entity_id,),
        )
        self.db.commit()

    def record_access(self, entity_id: int) -> None:
        """Record an access event: increment count and update last_access.

        Also records a daily entry in entity_access_log for consolidation tracking.
        """
        self.db.execute(
            """
            INSERT INTO entity_access (entity_id, access_count, last_access)
            VALUES (?, 1, datetime('now'))
            ON CONFLICT(entity_id) DO UPDATE SET
                access_count = access_count + 1,
                last_access = datetime('now')
            """,
            (entity_id,),
        )
        self.db.execute(
            """
            INSERT INTO entity_access_log (entity_id, access_date, access_count)
            VALUES (?, DATE('now'), 1)
            ON CONFLICT(entity_id, access_date) DO UPDATE SET
                access_count = access_count + 1
            """,
            (entity_id,),
        )
        self.db.commit()

    def get_access_data(self, entity_ids: list[int]) -> dict[int, dict]:
        """Get access data for a list of entity IDs.
        Returns dict with access_count and last_access for each.
        Defaults to access_count=0, last_access=created_at if no record exists."""
        if not entity_ids:
            return {}
        placeholders = ",".join("?" for _ in entity_ids)
        rows = self.db.execute(
            f"""
            SELECT e.id, e.created_at,
                   COALESCE(ea.access_count, 0) AS access_count,
                   COALESCE(ea.last_access, e.created_at) AS last_access
            FROM entities e
            LEFT JOIN entity_access ea ON ea.entity_id = e.id
            WHERE e.id IN ({placeholders})
            """,
            entity_ids,
        ).fetchall()
        return {
            r["id"]: {
                "access_count": r["access_count"],
                "last_access": r["last_access"],
            }
            for r in rows
        }

    def get_access_days(self, entity_ids: list[int]) -> dict[int, int]:
        """Get count of unique days with access for each entity.

        Returns {entity_id: access_days_count}.
        Entities with no rows in entity_access_log (pre-migration accesses)
        are not included in the result — callers should default to 1.
        """
        if not entity_ids:
            return {}
        placeholders = ",".join("?" for _ in entity_ids)
        rows = self.db.execute(
            f"""
            SELECT entity_id, COUNT(*) as access_days
            FROM entity_access_log
            WHERE entity_id IN ({placeholders})
            GROUP BY entity_id
            """,
            entity_ids,
        ).fetchall()
        return {r["entity_id"]: r["access_days"] for r in rows}

    def get_entity_degrees(self, entity_ids: list[int]) -> dict[int, int]:
        """Get relation degree count for a list of entity IDs.
        Counts relations where entity is either from_entity or to_entity."""
        if not entity_ids:
            return {}
        placeholders = ",".join("?" for _ in entity_ids)
        rows = self.db.execute(
            f"""
            SELECT entity_id, COUNT(*) AS degree FROM (
                SELECT from_entity AS entity_id FROM relations WHERE from_entity IN ({placeholders})
                UNION ALL
                SELECT to_entity AS entity_id FROM relations WHERE to_entity IN ({placeholders})
            ) GROUP BY entity_id
            """,
            entity_ids + entity_ids,
        ).fetchall()
        result = {eid: 0 for eid in entity_ids}
        for r in rows:
            result[r["entity_id"]] = r["degree"]
        return result

    # ------------------------------------------------------------------
    # Limbic: Co-occurrence tracking
    # ------------------------------------------------------------------

    def record_co_occurrences(self, entity_ids: list[int]) -> None:
        """Record co-occurrences for all unique pairs of entity IDs.
        Canonical ordering: entity_a_id < entity_b_id always."""
        if len(entity_ids) < 2:
            return
        sorted_ids = sorted(entity_ids)
        for i in range(len(sorted_ids)):
            for j in range(i + 1, len(sorted_ids)):
                a, b = sorted_ids[i], sorted_ids[j]
                self.db.execute(
                    """
                    INSERT INTO co_occurrences (entity_a_id, entity_b_id, co_count, last_co)
                    VALUES (?, ?, 1, datetime('now'))
                    ON CONFLICT(entity_a_id, entity_b_id) DO UPDATE SET
                        co_count = co_count + 1,
                        last_co = datetime('now')
                    """,
                    (a, b),
                )
        self.db.commit()

    def get_co_occurrences(
        self, entity_ids: list[int]
    ) -> dict[tuple[int, int], dict[str, Any]]:
        """Get co-occurrence counts for pairs within a set of entity IDs.
        Returns dict with (id_a, id_b) -> {"co_count": int, "last_co": str}, where id_a < id_b."""
        if len(entity_ids) < 2:
            return {}
        id_set = set(entity_ids)
        placeholders = ",".join("?" for _ in entity_ids)
        rows = self.db.execute(
            f"""
            SELECT entity_a_id, entity_b_id, co_count, last_co
            FROM co_occurrences
            WHERE entity_a_id IN ({placeholders}) AND entity_b_id IN ({placeholders})
            """,
            entity_ids + entity_ids,
        ).fetchall()
        return {
            (r["entity_a_id"], r["entity_b_id"]): {
                "co_count": r["co_count"],
                "last_co": r["last_co"],
            }
            for r in rows
            if r["entity_a_id"] in id_set and r["entity_b_id"] in id_set
        }

    # ------------------------------------------------------------------
    # Limbic: FTS5 full-text search
    # ------------------------------------------------------------------

    def _sync_fts(self, entity_id: int) -> None:
        """Rebuild FTS index entry for an entity from current DB state.
        Idempotent: INSERT OR REPLACE. No-op if FTS not available."""
        if not self._fts_available:
            return
        try:
            entity = self.get_entity_by_id(entity_id)
            if not entity:
                return
            obs_data = self.get_observations_with_ids(
                entity_id, exclude_superseded=True
            )
            obs_parts = []
            for o in obs_data:
                kind = o.get("kind", "generic")
                content = o["content"]
                if kind != "generic":
                    obs_parts.append(f"[{kind}] {content}")
                else:
                    obs_parts.append(content)
            obs_text = " | ".join(obs_parts)
            self.db.execute(
                "INSERT OR REPLACE INTO entity_fts(rowid, name, entity_type, obs_text) VALUES (?, ?, ?, ?)",
                (entity_id, entity["name"], entity["entity_type"], obs_text),
            )
            self.db.commit()
        except Exception as exc:
            logger.warning("FTS sync failed for entity %s: %s", entity_id, exc)

    def _delete_fts(self, entity_id: int) -> None:
        """Delete FTS index entry for an entity."""
        if not self._fts_available:
            return
        try:
            self.db.execute("DELETE FROM entity_fts WHERE rowid = ?", (entity_id,))
            self.db.commit()
        except Exception as exc:
            logger.warning("FTS delete failed for entity %s: %s", entity_id, exc)

    def _backfill_fts(self) -> None:
        """Populate FTS index from all existing entities. Called from init_db."""
        try:
            entities = self.db.execute(
                "SELECT id, name, entity_type FROM entities"
            ).fetchall()
            if not entities:
                return
            for e in entities:
                obs_data = self.get_observations_with_ids(
                    e["id"], exclude_superseded=True
                )
                obs_parts = []
                for o in obs_data:
                    kind = o.get("kind", "generic")
                    content = o["content"]
                    if kind != "generic":
                        obs_parts.append(f"[{kind}] {content}")
                    else:
                        obs_parts.append(content)
                obs_text = " | ".join(obs_parts)
                self.db.execute(
                    "INSERT OR REPLACE INTO entity_fts(rowid, name, entity_type, obs_text) VALUES (?, ?, ?, ?)",
                    (e["id"], e["name"], e["entity_type"], obs_text),
                )
            self.db.commit()
            logger.info("FTS index backfilled with %d entities.", len(entities))
        except Exception as exc:
            logger.error("FTS backfill failed: %s", exc)

    def search_fts(self, query: str, limit: int = 10) -> list[dict]:
        """FTS5 full-text search using BM25 ranking.

        Searches across name, entity_type, and obs_text columns.
        Returns list of {"entity_id": int, "rank": float} ordered by relevance (descending).
        Returns empty list if FTS not available or query is empty.
        """
        if not self._fts_available or not query.strip():
            return []
        try:
            # Escape FTS5 special characters: " * ( ) : ^
            # Wrap each token in double quotes for exact matching
            tokens = query.strip().split()
            escaped = " ".join(f'"{t}"' for t in tokens if t)
            if not escaped:
                return []

            rows = self.db.execute(
                """
                SELECT rowid AS entity_id, rank
                FROM entity_fts
                WHERE entity_fts MATCH ?
                ORDER BY rank
                LIMIT ?;
                """,
                (escaped, limit),
            ).fetchall()
            # FTS5 rank is negative (lower = better). Negate for conventional ordering.
            return [
                {"entity_id": r["entity_id"], "rank": -float(r["rank"])} for r in rows
            ]
        except Exception as exc:
            logger.warning("FTS5 search failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # A/B Testing: Search event logging
    # ------------------------------------------------------------------

    def log_search_event(
        self,
        query_text: str,
        treatment: int,
        k_limit: int,
        num_results: int,
        duration_ms: float | None,
        engine_used: str,
    ) -> int:
        """Log a search event. Returns the new event_id."""
        import hashlib

        query_hash = hashlib.md5(query_text.encode()).hexdigest()
        cursor = self.db.execute(
            """
            INSERT INTO search_events (query_text, query_hash, treatment, k_limit, num_results, duration_ms, engine_used)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                query_text,
                query_hash,
                treatment,
                k_limit,
                num_results,
                duration_ms,
                engine_used,
            ),
        )
        self.db.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def log_search_results(
        self,
        event_id: int,
        results: list[dict],
    ) -> None:
        """Log search results for an event.

        Uses INSERT OR IGNORE to handle UNIQUE constraint on (event_id, entity_id).
        Skips results that already have an entry for the same event_id+entity_id pair.
        """
        for r in results:
            self.db.execute(
                """
                INSERT OR IGNORE INTO search_results
                (event_id, entity_id, entity_name, rank, limbic_score, cosine_sim, importance, temporal, cooc_boost, baseline_rank)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    r["entity_id"],
                    r["entity_name"],
                    r["rank"],
                    r.get("limbic_score"),
                    r.get("cosine_sim"),
                    r.get("importance"),
                    r.get("temporal"),
                    r.get("cooc_boost"),
                    r.get("baseline_rank"),
                ),
            )
        self.db.commit()

    def log_implicit_feedback(
        self,
        event_id: int,
        entity_id: int,
        re_accessed: bool,
        access_delta: int | None,
        session_id: str | None,
    ) -> None:
        """Log implicit feedback for a search result."""
        self.db.execute(
            """
            INSERT INTO implicit_feedback (event_id, entity_id, re_accessed, access_delta, session_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (event_id, entity_id, int(re_accessed), access_delta, session_id),
        )
        self.db.commit()

    def update_search_event_completion(
        self,
        event_id: int,
        num_results: int,
        duration_ms: float,
        engine_used: str,
    ) -> None:
        """Update num_results, duration_ms and engine_used for an existing search event."""
        try:
            self.db.execute(
                """
                UPDATE search_events
                SET num_results = ?, duration_ms = ?, engine_used = ?
                WHERE event_id = ?
                """,
                (num_results, duration_ms, engine_used, event_id),
            )
            self.db.commit()
        except Exception as exc:
            self.db.rollback()
            logger.error(
                "Failed to update search event %s completion: %s", event_id, exc
            )

    # ------------------------------------------------------------------
    # Phase 4: Reflections CRUD
    # ------------------------------------------------------------------

    def add_reflection(
        self,
        target_type: str,
        target_id: int | None,
        author: str,
        content: str,
        mood: str | None = None,
    ) -> dict | None:
        """Add a narrative reflection. Returns the reflection dict or None on error.

        Validates target_type, author, and mood.
        If target_type is 'entity' or 'relation', target_id is required.
        If target_type is 'global', target_id must be None.
        Generates embedding and syncs FTS.
        """
        # Validations
        if target_type not in VALID_TARGET_TYPES:
            logger.warning("Invalid target_type: %s", target_type)
            return None
        if author not in VALID_AUTHORS:
            logger.warning("Invalid author: %s", author)
            return None
        if mood is not None and mood not in VALID_MOODS:
            logger.warning("Invalid mood: %s", mood)
            return None
        if target_type in ("entity", "session", "relation") and target_id is None:
            logger.warning("target_id required for target_type '%s'", target_type)
            return None
        if target_type == "global" and target_id is not None:
            logger.warning("target_id must be None for global reflections")
            return None

        # Insert reflection
        cursor = self.db.execute(
            "INSERT INTO reflections (target_type, target_id, author, content, mood) VALUES (?, ?, ?, ?, ?)",
            (target_type, target_id, author, content, mood),
        )
        self.db.commit()
        reflection_id = cursor.lastrowid

        # Sync FTS (manual content sync)
        if self._fts_available:
            try:
                self.db.execute(
                    "INSERT INTO reflection_fts(rowid, content, author, mood) VALUES (?, ?, ?, ?)",
                    (reflection_id, content, author, mood or ""),
                )
                self.db.commit()
            except Exception as exc:
                logger.warning(
                    "Failed to sync reflection FTS for id %s: %s", reflection_id, exc
                )

        # Generate embedding
        engine = self._get_embedding_engine()
        if engine and engine.available:
            try:
                from mcp_memory.embeddings import serialize_f32

                vector = engine.encode([content])
                embedding_bytes = serialize_f32(vector[0])
                self.db.execute(
                    "INSERT INTO reflection_embeddings(rowid, embedding) VALUES (?, ?)",
                    (reflection_id, embedding_bytes),
                )
                self.db.commit()
            except Exception as exc:
                logger.warning(
                    "Failed to store reflection embedding for id %s: %s",
                    reflection_id,
                    exc,
                )

        return {
            "id": reflection_id,
            "target_type": target_type,
            "target_id": target_id,
            "author": author,
            "content": content,
            "mood": mood,
            "created_at": self.db.execute(
                "SELECT created_at FROM reflections WHERE id = ?", (reflection_id,)
            ).fetchone()["created_at"],
        }

    def get_reflections_for_target(
        self, target_type: str, target_id: int | None = None
    ) -> list[dict]:
        """Get reflections for a specific target (entity, session, relation, or global)."""
        if target_type == "global":
            rows = self.db.execute(
                "SELECT id, author, content, mood, created_at FROM reflections WHERE target_type = 'global' ORDER BY created_at"
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT id, author, content, mood, created_at FROM reflections WHERE target_type = ? AND target_id = ? ORDER BY created_at",
                (target_type, target_id),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "author": r["author"],
                "content": r["content"],
                "mood": r["mood"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def search_reflection_fts(self, query: str, limit: int = 10) -> list[dict]:
        """FTS5 search on reflection_fts. Returns list of {id, rank}."""
        if not self._fts_available or not query.strip():
            return []
        try:
            tokens = query.strip().split()
            escaped = " ".join(f'"{t}"' for t in tokens if t)
            if not escaped:
                return []
            rows = self.db.execute(
                """
                SELECT rowid AS id, rank
                FROM reflection_fts
                WHERE reflection_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (escaped, limit),
            ).fetchall()
            return [{"id": r["id"], "rank": -float(r["rank"])} for r in rows]
        except Exception as exc:
            logger.warning("Reflection FTS search failed: %s", exc)
            return []

    def search_reflection_embeddings(
        self, query_embedding: bytes, limit: int = 10
    ) -> list[dict]:
        """KNN search on reflection_embeddings. Returns list of {id, distance}."""
        if not self._vec_loaded:
            return []
        try:
            rows = self.db.execute(
                """
                SELECT rowid AS id, distance
                FROM reflection_embeddings
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT ?
                """,
                (query_embedding, limit),
            ).fetchall()
            return [{"id": r["id"], "distance": float(r["distance"])} for r in rows]
        except Exception as exc:
            logger.warning("Reflection embedding search failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Consolidation report data (read-only queries)
    # ------------------------------------------------------------------

    def get_consolidation_data(self, stale_days: float = 90.0) -> dict[str, Any]:
        """Collect data for a consolidation report. All queries are read-only.

        Args:
            stale_days: Number of days threshold for considering an entity stale.

        Returns:
            Dict with total_entities, total_observations, flagged_observations,
            stale_entities, and entity_sizes.
        """
        data: dict[str, Any] = {}

        # --- 1. Totals ---
        data["total_entities"] = self.db.execute(
            "SELECT COUNT(*) FROM entities"
        ).fetchone()[0]

        data["total_observations"] = self.db.execute(
            "SELECT COUNT(*) FROM observations"
        ).fetchone()[0]

        # --- 2. Flagged observations (similarity_flag = 1) ---
        rows = self.db.execute(
            """
            SELECT e.name, o.id, o.content, o.similarity_flag
            FROM observations o
            JOIN entities e ON e.id = o.entity_id
            WHERE o.similarity_flag = 1
            ORDER BY e.name, o.id
            """
        ).fetchall()

        flagged: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            name = r["name"]
            if name not in flagged:
                flagged[name] = []
            flagged[name].append(
                {
                    "id": r["id"],
                    "content": r["content"],
                    "similarity_flag": r["similarity_flag"],
                }
            )
        data["flagged_observations"] = flagged

        # --- 3. Stale entities ---
        # Entities with observations but not accessed in N days and low access count.
        # LEFT JOIN entity_access so entities without access records are included.
        # COALESCE handles NULLs from missing entity_access rows.
        stale_param = str(int(stale_days))
        rows = self.db.execute(
            """
            SELECT e.name, e.entity_type, e.status, e.created_at,
                   COALESCE(ea.access_count, 0) AS access_count,
                   ea.last_access,
                   COUNT(o.id) AS obs_count
            FROM entities e
            LEFT JOIN entity_access ea ON ea.entity_id = e.id
            JOIN observations o ON o.entity_id = e.id
            GROUP BY e.id
            HAVING (ea.last_access IS NULL
                    OR datetime(ea.last_access) < datetime('now', '-' || ? || ' days'))
               AND COALESCE(ea.access_count, 0) <= 2
               AND e.status != 'archivado'
            ORDER BY CASE WHEN ea.last_access IS NULL THEN 0 ELSE 1 END,
                     ea.last_access ASC
            """,
            (stale_param,),
        ).fetchall()

        data["stale_entities"] = [
            {
                "entity_name": r["name"],
                "entity_type": r["entity_type"],
                "status": r["status"],
                "created_at": r["created_at"],
                "access_count": r["access_count"],
                "last_access": r["last_access"],
                "observation_count": r["obs_count"],
            }
            for r in rows
        ]

        # --- 4. Entity sizes ---
        rows = self.db.execute(
            """
            SELECT e.name, COUNT(o.id) AS obs_count
            FROM entities e
            LEFT JOIN observations o ON o.entity_id = e.id
            GROUP BY e.id
            ORDER BY obs_count DESC
            """
        ).fetchall()

        data["entity_sizes"] = {r["name"]: r["obs_count"] for r in rows}

        return data
