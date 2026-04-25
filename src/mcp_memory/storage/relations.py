"""Relation CRUD operations for the MCP Memory knowledge graph."""

import sqlite3
import logging

from mcp_memory.retry import retry_on_locked
from mcp_memory.storage._constants import INVERSE_RELATIONS

logger = logging.getLogger(__name__)


class RelationsMixin:
    """Relation CRUD methods for MemoryStore."""

    @retry_on_locked
    def create_relation(
        self,
        from_entity_id: int,
        to_entity_id: int,
        relation_type: str,
        context: str | None = None,
        *,
        auto_commit: bool = True,
    ) -> bool:
        """INSERT relation. ON CONFLICT -> return False (already exists).
        If relation_type has an inverse, automatically creates the inverse relation."""
        with self._write_lock:
            try:
                self.db.execute(
                    """
                    INSERT INTO relations (from_entity, to_entity, relation_type, context)
                    VALUES (?, ?, ?, ?);
                    """,
                    (from_entity_id, to_entity_id, relation_type, context),
                )
                if auto_commit:
                    self.db.commit()
                # Auto-create inverse relation (best-effort)
                self._ensure_inverse_relation(
                    from_entity_id, to_entity_id, relation_type, context,
                    auto_commit=auto_commit,
                )
                return True
            except sqlite3.IntegrityError:
                # UNIQUE constraint violated -> already exists
                return False

    @retry_on_locked
    def _ensure_inverse_relation(
        self,
        from_id: int,
        to_id: int,
        relation_type: str,
        context: str | None = None,
        *,
        auto_commit: bool = True,
    ) -> None:
        """Auto-create the inverse relation if one is defined in INVERSE_RELATIONS.
        Best-effort: silently skips if inverse already exists or type has no inverse."""
        with self._write_lock:
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
                if auto_commit:
                    self.db.commit()
            except sqlite3.IntegrityError:
                # Race condition or already exists — ignore
                pass

    @retry_on_locked
    def _end_relation(self, relation_id: int) -> bool:
        """Deactivate a relation by setting active=0 and ended_at.
        Returns True if the relation was found and updated, False otherwise."""
        with self._write_lock:
            cursor = self.db.execute(
                "UPDATE relations SET active = 0, ended_at = datetime('now') WHERE id = ?",
                (relation_id,),
            )
            self.db.commit()
            return cursor.rowcount > 0

    def get_relation_by_id(self, relation_id: int) -> dict | None:
        """Get a single relation by its ID. Returns dict or None."""
        with self._write_lock:
            row = self.db.execute(
                "SELECT id, from_entity, to_entity, relation_type, context, active, ended_at, created_at "
                "FROM relations WHERE id = ?",
                (relation_id,),
            ).fetchone()
        if row is None:
            return None
        return dict(row)

    @retry_on_locked
    def delete_relation(
        self, from_entity_id: int, to_entity_id: int, relation_type: str
    ) -> bool:
        """DELETE relation. Returns True if it existed."""
        with self._write_lock:
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
        with self._write_lock:
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
        with self._write_lock:
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

    def find_inverse_relation(self, from_entity_id: int, to_entity_id: int, relation_type: str) -> dict | None:
        """Find an active inverse relation by swapped entities and inverse type.

        Returns the relation dict with id and active status if found and active,
        otherwise None.
        """
        with self._write_lock:
            row = self.db.execute(
                "SELECT id, active FROM relations WHERE from_entity = ? AND to_entity = ? AND relation_type = ?",
                (to_entity_id, from_entity_id, relation_type),
            ).fetchone()
        if row and row["active"] == 1:
            return dict(row)
        return None

    def get_relations_for_entity_batch(self, entity_ids: list[int]) -> dict[int, list[dict]]:
        """Get all relations for multiple entities in a single query.

        Returns dict mapping entity_id -> list of relation dicts with
        relation_type, target_name, direction, context, active, ended_at.
        """
        if not entity_ids:
            return {}
        with self._write_lock:
            placeholders = ",".join("?" for _ in entity_ids)
            rows = self.db.execute(
                f"""
                SELECT r.id, r.from_entity, r.to_entity, r.relation_type,
                       r.context, r.active, r.ended_at,
                       e_from.name AS from_name, e_to.name AS to_name
                FROM relations r
                JOIN entities e_from ON r.from_entity = e_from.id
                JOIN entities e_to ON r.to_entity = e_to.id
                WHERE r.from_entity IN ({placeholders}) OR r.to_entity IN ({placeholders})
                """,
                entity_ids + entity_ids,
            ).fetchall()
        result = {eid: [] for eid in entity_ids}
        for r in rows:
            if r["from_entity"] in result:
                result[r["from_entity"]].append(
                    {
                        "relation_type": r["relation_type"],
                        "target_name": r["to_name"],
                        "direction": "from",
                        "context": r["context"],
                        "active": r["active"],
                        "ended_at": r["ended_at"],
                    }
                )
            if r["to_entity"] in result:
                result[r["to_entity"]].append(
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
