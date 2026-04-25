"""Reflection CRUD and search for the MCP Memory knowledge graph."""

import logging

from mcp_memory.retry import retry_on_locked
from mcp_memory.storage._constants import VALID_TARGET_TYPES, VALID_AUTHORS, VALID_MOODS

logger = logging.getLogger(__name__)


class ReflectionsMixin:
    """Reflection CRUD and search methods for MemoryStore."""

    # ------------------------------------------------------------------
    # Phase 4: Reflections CRUD
    # ------------------------------------------------------------------

    @retry_on_locked
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
        with self._write_lock:
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
            if target_type in ("entity", "relation") and target_id is None:
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
        with self._write_lock:
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

    def get_reflections_for_target_batch(
        self, target_type: str, target_ids: list[int]
    ) -> dict[int, list[dict]]:
        """Get reflections for multiple targets in a single query.

        Returns dict mapping target_id -> list of reflection dicts.
        """
        if not target_ids or target_type == "global":
            return {}
        with self._write_lock:
            placeholders = ",".join("?" for _ in target_ids)
            rows = self.db.execute(
                f"SELECT id, target_id, author, content, mood, created_at FROM reflections "
                f"WHERE target_type = ? AND target_id IN ({placeholders}) ORDER BY created_at",
                (target_type, *target_ids),
            ).fetchall()
        result = {tid: [] for tid in target_ids}
        for r in rows:
            result[r["target_id"]].append(
                {
                    "id": r["id"],
                    "author": r["author"],
                    "content": r["content"],
                    "mood": r["mood"],
                    "created_at": r["created_at"],
                }
            )
        return result

    def search_reflection_fts(self, query: str, limit: int = 10) -> list[dict]:
        """FTS5 search on reflection_fts. Returns list of {id, rank}."""
        if not self._fts_available or not query.strip():
            return []
        try:
            with self._write_lock:
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
            with self._write_lock:
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

    def search_reflections_filtered(
        self,
        candidate_ids: list[int],
        author: str | None = None,
        mood: str | None = None,
        target_type: str | None = None,
    ) -> list[dict]:
        """Fetch reflection data for candidate IDs and apply optional filters.

        Returns list of dicts with id, target_type, target_id, author, content,
        mood, and created_at.
        """
        if not candidate_ids:
            return []

        placeholders = ",".join("?" for _ in candidate_ids)
        query_sql = f"""
            SELECT r.id, r.target_type, r.target_id, r.author, r.content, r.mood, r.created_at
            FROM reflections r
            WHERE r.id IN ({placeholders})
        """
        params: list[Any] = list(candidate_ids)

        conditions = []
        if author:
            conditions.append("r.author = ?")
            params.append(author)
        if mood:
            conditions.append("r.mood = ?")
            params.append(mood)
        if target_type:
            conditions.append("r.target_type = ?")
            params.append(target_type)

        if conditions:
            query_sql += " AND " + " AND ".join(conditions)

        with self._write_lock:
            rows = self.db.execute(query_sql, params).fetchall()
        return [dict(r) for r in rows]
