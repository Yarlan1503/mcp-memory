"""Relation and migration tools for MCP Memory knowledge graph."""

import logging
from typing import Any

import mcp_memory.server as _server_mod
from mcp_memory._helpers import _get_engine

logger = logging.getLogger(__name__)


def migrate(
    source_path: str = "",
) -> dict[str, Any]:
    """Migrate data from Anthropic MCP Memory JSONL format to SQLite.
    This is idempotent — running it multiple times won't duplicate data.
    Requires a valid source_path to an existing JSONL file."""
    if not source_path:
        return {
            "error": "source_path is required — provide the path to an Anthropic MCP Memory JSONL file"
        }
    try:
        store = _server_mod.store
        from mcp_memory.migrate import migrate_jsonl

        engine = _get_engine()
        result = migrate_jsonl(store, source_path, engine)
        return result
    except Exception as e:
        logger.error("Error in migrate: %s", e)
        return {"error": str(e)}


def end_relation(relation_id: int) -> dict[str, Any]:
    """Expire an active relation by setting active=0 and ended_at=now.

    For inverse pairs (contiene↔parte_de), also expires the inverse relation.

    Args:
        relation_id: The ID of the relation to expire.
    """
    try:
        store = _server_mod.store
        from mcp_memory.storage import INVERSE_RELATIONS

        # 1. Look up the relation
        rel = store.get_relation_by_id(relation_id)
        if rel is None:
            return {"error": f"Relation with id={relation_id} not found"}

        # 2. Check if already inactive
        if rel["active"] == 0:
            # Already expired — return notice, not error
            return {
                "notice": f"Relation {relation_id} is already inactive",
                "relation": {
                    "id": rel["id"],
                    "from_entity": rel["from_entity"],
                    "to_entity": rel["to_entity"],
                    "relation_type": rel["relation_type"],
                    "active": bool(rel["active"]),
                    "ended_at": rel["ended_at"],
                },
            }

        # 3. Expire the relation
        success = store._end_relation(relation_id)
        if not success:
            return {"error": f"Failed to expire relation {relation_id}"}

        # 4. Check for inverse and expire it too
        inverse_expired = None
        inverse_type = INVERSE_RELATIONS.get(rel["relation_type"])
        if inverse_type:
            # Find the inverse: (to_entity, from_entity, inverse_type)
            inv_row = store.db.execute(
                "SELECT id, active FROM relations WHERE from_entity = ? AND to_entity = ? AND relation_type = ?",
                (rel["to_entity"], rel["from_entity"], inverse_type),
            ).fetchone()
            if inv_row and inv_row["active"] == 1:
                store._end_relation(inv_row["id"])
                inverse_expired = inv_row["id"]

        # 5. Re-fetch to get updated ended_at
        updated = store.get_relation_by_id(relation_id)

        # Resolve entity names for the response
        from_ent = store.get_entity_by_id(rel["from_entity"])
        to_ent = store.get_entity_by_id(rel["to_entity"])

        result = {
            "relation": {
                "id": updated["id"],
                "from": from_ent["name"] if from_ent else str(rel["from_entity"]),
                "to": to_ent["name"] if to_ent else str(rel["to_entity"]),
                "relation_type": updated["relation_type"],
                "active": bool(updated["active"]),
                "ended_at": updated["ended_at"],
                "context": updated["context"],
            }
        }
        if inverse_expired is not None:
            result["inverse_expired_id"] = inverse_expired

        return result

    except Exception as e:
        logger.error("Error in end_relation: %s", e)
        return {"error": str(e)}
