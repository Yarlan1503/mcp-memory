"""Shared helpers for MCP Memory tools."""

import functools
import logging
from typing import Any

logger = logging.getLogger(__name__)


def tool_error_handler(func):
    """Decorator that wraps tool functions with standard error handling.

    Catches exceptions, logs them with logger.error, and returns {"error": str(e)}.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error("Error in %s: %s", func.__name__, e)
            return {"error": str(e)}

    return wrapper


def _get_engine() -> Any:
    """Get embedding engine instance (may be unavailable).

    Lazy import to allow the server to start even when
    ``mcp_memory.embeddings`` hasn't been created yet (T6).
    """
    try:
        from mcp_memory.embeddings import EmbeddingEngine

        return EmbeddingEngine.get_instance()
    except Exception:
        return None


def _entity_to_output(
    row: dict,
    observations: list[str] | list[dict],
    relations: list[dict] | None = None,
) -> dict:
    """Convert DB row + observations to EntityOutput dict.
    Optionally includes relations with context, active, and ended_at."""
    obs_list = observations
    output = {
        "name": row["name"],
        "entityType": row["entity_type"],
        "status": row.get("status", "activo"),
        "observations": obs_list,
    }
    if relations is not None:
        output["relations"] = relations
    return output


def _recompute_embedding(entity_id: int, name: str, entity_type: str) -> None:
    """Recompute and store embedding for an entity (if engine available)."""
    # Import inside function to avoid circular import issues
    import mcp_memory.server as _server_mod

    store = _server_mod.store
    engine = _get_engine()
    if not engine or not getattr(engine, "available", False):
        return
    try:
        from mcp_memory.embeddings import serialize_f32

        # Get entity status
        entity_data = store.get_entity_by_id(entity_id)
        status = entity_data.get("status", "activo") if entity_data else "activo"

        # Get observations with IDs (for kind info)
        obs_data = store.get_observations_with_ids(entity_id, exclude_superseded=True)
        relations = store.get_relations_for_entity(entity_id)
        text = engine.prepare_entity_text(
            name, entity_type, obs_data, relations, status=status
        )
        vector = engine.encode([text])
        store.store_embedding(entity_id, serialize_f32(vector[0]))
    except Exception as e:
        logger.warning("Failed to recompute embedding for %s: %s", name, e)
