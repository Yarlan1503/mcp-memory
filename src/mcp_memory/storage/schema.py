"""Schema initialization and migrations for the MCP Memory knowledge graph."""

import logging
import sqlite3

from mcp_memory.retry import retry_on_locked
from mcp_memory.storage._constants import LEGACY_RELATION_TYPES

logger = logging.getLogger(__name__)


class SchemaMixin:
    """Schema initialization and idempotent migrations."""

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
        self._add_observation_superseded_index()

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

    def _add_observation_superseded_index(self) -> None:
        """Idempotent migration: create composite index on observations(entity_id, superseded_at)."""
        try:
            self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_obs_entity_superseded "
                "ON observations(entity_id, superseded_at)"
            )
            self.db.commit()
            logger.info("Composite index idx_obs_entity_superseded verified.")
        except Exception as exc:
            logger.warning("Failed to create composite index: %s", exc)
