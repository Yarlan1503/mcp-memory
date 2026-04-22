"""Entity and observation CRUD operations for the MCP Memory knowledge graph."""

import logging
import sqlite3

from mcp_memory.retry import retry_on_locked

logger = logging.getLogger(__name__)


class CoreMixin:
    """Entity and observation CRUD methods for MemoryStore."""

    # ------------------------------------------------------------------
    # Entity CRUD
    # ------------------------------------------------------------------

    @retry_on_locked
    def upsert_entity(self, name: str, entity_type: str, status: str = "activo", *, auto_commit: bool = True) -> int:
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
        if auto_commit:
            self.db.commit()
        row = self.db.execute(
            "SELECT id FROM entities WHERE name = ?", (name,)
        ).fetchone()
        entity_id = row["id"]  # type: ignore[return-value]
        if auto_commit:
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
        """All entities with their observations. Single query via LEFT JOIN."""
        rows = self.db.execute(
            "SELECT e.id, e.name, e.entity_type, e.status, e.created_at, e.updated_at, "
            "o.content "
            "FROM entities e "
            "LEFT JOIN observations o ON o.entity_id = e.id AND o.superseded_at IS NULL "
            "ORDER BY e.id, o.id"
        ).fetchall()

        entities_map: dict[int, dict] = {}
        for row in rows:
            eid = row["id"]
            if eid not in entities_map:
                entities_map[eid] = {
                    "id": eid,
                    "name": row["name"],
                    "entity_type": row["entity_type"],
                    "status": row["status"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "observations": [],
                }
            if row["content"] is not None:
                entities_map[eid]["observations"].append(row["content"])

        return list(entities_map.values())

    @retry_on_locked
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
            # 1. Delete embeddings (vec0 has no CASCADE support)
            if self._vec_loaded:
                try:
                    self.db.execute(
                        f"DELETE FROM entity_embeddings WHERE rowid IN ({id_placeholders})",
                        ids,
                    )
                except Exception:
                    logger.warning("Could not delete embeddings for entities %s", ids)

            # 2. Delete FTS entries (FTS5 doesn't CASCADE either)
            if self._fts_available:
                try:
                    self.db.execute(
                        f"DELETE FROM entity_fts WHERE rowid IN ({id_placeholders})",
                        ids,
                    )
                except Exception:
                    logger.warning("Could not delete FTS entries for entities %s", ids)

            # 3. Delete entities (CASCADE takes care of observations & relations)
            self.db.execute(
                f"DELETE FROM entities WHERE id IN ({id_placeholders})", ids
            )

            return len(ids)

    def search_entities(self, query: str) -> list[dict]:
        """
        LIKE search on name, entity_type, and observations.content.
        Returns entities with their observations.
        Uses 2 queries instead of N+1.
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

        if not rows:
            return []

        entity_ids = [r["id"] for r in rows]
        obs_map = self.get_observations_batch(entity_ids)

        return [{**dict(r), "observations": obs_map.get(r["id"], [])} for r in rows]

    # ------------------------------------------------------------------
    # Observation CRUD
    # ------------------------------------------------------------------

    @retry_on_locked
    def add_observations(
        self,
        entity_id: int,
        observations: list[str],
        kind: str = "generic",
        supersedes: int | None = None,
        *,
        auto_commit: bool = True,
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

        # Acquire write lock early to prevent contention during ONNX inference
        # Skip if already in a transaction (e.g., called from execute_entity_split)
        if auto_commit:
            try:
                self.db.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError:
                pass  # Already in a transaction — use existing lock

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

        def _dedup_observations(obs_list: list[str]) -> list[str]:
            """Remove observations already in DB for this entity and internal duplicates."""
            placeholders = ",".join("?" for _ in obs_list)
            rows = self.db.execute(
                f"SELECT content FROM observations WHERE entity_id = ? AND content IN ({placeholders})",
                (entity_id, *obs_list),
            ).fetchall()
            existing_set = {r["content"] for r in rows}
            seen: set[str] = set()
            new_obs: list[str] = []
            for content in obs_list:
                if content not in existing_set and content not in seen:
                    seen.add(content)
                    new_obs.append(content)
            return new_obs

        def _insert_many(obs_list: list[str]) -> int:
            """Batch insert observations with flag=0."""
            if not obs_list:
                return 0
            self.db.executemany(
                "INSERT INTO observations (entity_id, content, similarity_flag, kind) "
                "VALUES (?, ?, 0, ?)",
                [(entity_id, content, kind) for content in obs_list],
            )
            return len(obs_list)

        # Try to get embedding engine for semantic dedup
        engine = self._get_embedding_engine()
        existing_obs = self.get_observations(entity_id)

        if existing_obs and engine is not None:
            # --- Semantic dedup path ---
            try:
                import numpy as np  # noqa: F401

                # Batch encode all existing observations
                existing_embeddings = engine.encode(existing_obs)

                # Batch exact dedup before encoding new observations
                candidates = _dedup_observations(observations)

                if candidates:
                    candidate_embeddings = engine.encode(candidates)

                    for content, new_embedding in zip(candidates, candidate_embeddings):
                        # Cosine similarity with all existing (L2-normalised → dot product)
                        similarities = existing_embeddings @ new_embedding  # (n,)
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
                # Fallback: dedup exact + batch insert remaining without flag
                new_obs = _dedup_observations(observations)
                inserted += _insert_many(new_obs)
        else:
            # --- Normal path: no engine or no existing observations ---
            new_obs = _dedup_observations(observations)
            inserted += _insert_many(new_obs)

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
            if auto_commit:
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

    def get_observations_batch(
        self, entity_ids: list[int], exclude_superseded: bool = True
    ) -> dict[int, list[str]]:
        """Get observations for multiple entities in a single query.
        Returns dict mapping entity_id -> list of observation content strings."""
        if not entity_ids:
            return {}
        placeholders = ",".join("?" for _ in entity_ids)
        if exclude_superseded:
            rows = self.db.execute(
                f"SELECT entity_id, content FROM observations "
                f"WHERE entity_id IN ({placeholders}) AND superseded_at IS NULL "
                f"ORDER BY entity_id, id",
                entity_ids,
            ).fetchall()
        else:
            rows = self.db.execute(
                f"SELECT entity_id, content FROM observations "
                f"WHERE entity_id IN ({placeholders}) "
                f"ORDER BY entity_id, id",
                entity_ids,
            ).fetchall()

        result: dict[int, list[str]] = {}
        for row in rows:
            result.setdefault(row["entity_id"], []).append(row["content"])
        return result

    def get_observations_with_ids_batch(
        self, entity_ids: list[int], exclude_superseded: bool = True
    ) -> dict[int, list[dict]]:
        """Get full observation dicts for multiple entities in a single query.
        Returns dict mapping entity_id -> list of {id, content, similarity_flag,
        kind, supersedes, superseded_at}."""
        if not entity_ids:
            return {}
        placeholders = ",".join("?" for _ in entity_ids)
        if exclude_superseded:
            rows = self.db.execute(
                f"SELECT id, entity_id, content, similarity_flag, kind, supersedes, superseded_at "
                f"FROM observations "
                f"WHERE entity_id IN ({placeholders}) AND superseded_at IS NULL "
                f"ORDER BY entity_id, id",
                entity_ids,
            ).fetchall()
        else:
            rows = self.db.execute(
                f"SELECT id, entity_id, content, similarity_flag, kind, supersedes, superseded_at "
                f"FROM observations "
                f"WHERE entity_id IN ({placeholders}) "
                f"ORDER BY entity_id, id",
                entity_ids,
            ).fetchall()

        result: dict[int, list[dict]] = {}
        for r in rows:
            eid = r["entity_id"]
            result.setdefault(eid, []).append(
                {
                    "id": r["id"],
                    "content": r["content"],
                    "similarity_flag": r["similarity_flag"],
                    "kind": r["kind"] if r["kind"] is not None else "generic",
                    "supersedes": r["supersedes"],
                    "superseded_at": r["superseded_at"],
                }
            )
        return result

    def get_entities_batch(self, entity_ids: list[int]) -> dict[int, dict]:
        """Get entity data for multiple IDs in a single query.
        Returns dict mapping entity_id -> {id, name, entity_type, status,
        created_at, updated_at}."""
        if not entity_ids:
            return {}
        placeholders = ",".join("?" for _ in entity_ids)
        rows = self.db.execute(
            f"SELECT id, name, entity_type, status, created_at, updated_at "
            f"FROM entities WHERE id IN ({placeholders})",
            entity_ids,
        ).fetchall()
        return {r["id"]: dict(r) for r in rows}

    @retry_on_locked
    def delete_observations(self, entity_id: int, observations: list[str], *, auto_commit: bool = True) -> int:
        """DELETE by exact content match. Returns count deleted."""
        if not observations:
            return 0
        placeholders = ",".join("?" for _ in observations)
        cursor = self.db.execute(
            f"DELETE FROM observations WHERE entity_id = ? AND content IN ({placeholders})",
            [entity_id, *observations],
        )
        if auto_commit:
            self.db.commit()
            self._sync_fts(entity_id)
        return cursor.rowcount
