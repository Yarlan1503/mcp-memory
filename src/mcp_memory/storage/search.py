"""FTS5 search, embedding operations, and search event logging for the MCP Memory knowledge graph."""

import logging

from mcp_memory.retry import retry_on_locked

logger = logging.getLogger(__name__)


class SearchMixin:
    """FTS search, embedding, and search event logging methods for MemoryStore."""

    # ------------------------------------------------------------------
    # Embedding ops
    # ------------------------------------------------------------------

    @retry_on_locked
    def store_embedding(self, entity_id: int, embedding: bytes) -> None:
        """DELETE + INSERT embedding (vec0 doesn't support OR REPLACE). entity_id is rowid in vec0."""
        with self._write_lock:
            if not self._vec_loaded:
                logger.warning("sqlite-vec not loaded — cannot store embedding.")
                return
            try:
                self.db.execute(
                    "DELETE FROM entity_embeddings WHERE rowid = ?",
                    (entity_id,),
                )
                self.db.execute(
                    "INSERT INTO entity_embeddings(rowid, embedding) VALUES (?, ?)",
                    (entity_id, embedding),
                )
                self.db.commit()
            except Exception as exc:
                self.db.rollback()
                logger.error("Failed to store embedding for entity %s: %s", entity_id, exc)

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
    # Limbic: FTS5 full-text search
    # ------------------------------------------------------------------

    def _format_obs_for_fts(self, obs_data: list[dict]) -> str:
        """Format observation list into a single FTS text string."""
        obs_parts = []
        for o in obs_data:
            kind = o.get("kind", "generic")
            content = o["content"]
            if kind != "generic":
                obs_parts.append(f"[{kind}] {content}")
            else:
                obs_parts.append(content)
        return " | ".join(obs_parts)

    @retry_on_locked
    def _sync_fts(self, entity_id: int, *, auto_commit: bool = True) -> None:
        """Rebuild FTS index entry for an entity from current DB state.
        Idempotent: INSERT OR REPLACE. No-op if FTS not available."""
        with self._write_lock:
            if not self._fts_available:
                return
            try:
                entity = self.get_entity_by_id(entity_id)
                if not entity:
                    return
                obs_data = self.get_observations_with_ids(
                    entity_id, exclude_superseded=True
                )
                obs_text = self._format_obs_for_fts(obs_data)
                self.db.execute(
                    "INSERT OR REPLACE INTO entity_fts(rowid, name, entity_type, obs_text) VALUES (?, ?, ?, ?)",
                    (entity_id, entity["name"], entity["entity_type"], obs_text),
                )
                if auto_commit:
                    self.db.commit()
            except Exception as exc:
                logger.warning("FTS sync failed for entity %s: %s", entity_id, exc)

    @retry_on_locked
    def _backfill_fts(self) -> None:
        """Populate FTS index from all existing entities. Called from init_db."""
        with self._write_lock:
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
                    obs_text = self._format_obs_for_fts(obs_data)
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

    @retry_on_locked
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
        with self._write_lock:
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

    @retry_on_locked
    def log_search_results(
        self,
        event_id: int,
        results: list[dict],
    ) -> None:
        """Log search results for an event.

        Uses INSERT OR IGNORE to handle UNIQUE constraint on (event_id, entity_id).
        Skips results that already have an entry for the same event_id+entity_id pair.
        """
        with self._write_lock:
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

    @retry_on_locked
    def update_search_event_completion(
        self,
        event_id: int,
        num_results: int,
        duration_ms: float,
        engine_used: str,
    ) -> None:
        """Update num_results, duration_ms and engine_used for an existing search event."""
        with self._write_lock:
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
