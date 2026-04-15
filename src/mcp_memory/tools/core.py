"""Core CRUD tools for MCP Memory knowledge graph."""

import logging
from typing import Any

import mcp_memory.server as _server_mod
from mcp_memory.config import (
    MAX_ENTITIES_PER_CALL,
    MAX_OBSERVATION_LENGTH,
    MAX_OBSERVATIONS_PER_CALL,
)
from mcp_memory.models import EntityInput, RelationInput
from mcp_memory._helpers import _recompute_embedding

logger = logging.getLogger(__name__)


# ============================================================
# TOOL: create_entities
# ============================================================
def create_entities(entities: list[dict[str, Any]]) -> dict[str, Any]:
    """Create or update entities in the knowledge graph.
    If an entity already exists, merge observations (don't overwrite).
    Returns the created/updated entities."""
    try:
        store = _server_mod.store
        # Input validation
        if len(entities) > MAX_ENTITIES_PER_CALL:
            return {
                "error": f"Too many entities: {len(entities)} > {MAX_ENTITIES_PER_CALL}. Create fewer entities at a time."
            }
        for i, entity_dict in enumerate(entities):
            obs_list = entity_dict.get("observations", [])
            if len(obs_list) > MAX_OBSERVATIONS_PER_CALL:
                return {
                    "error": f"Entity {i} has too many observations: {len(obs_list)} > {MAX_OBSERVATIONS_PER_CALL}. Split into smaller batches."
                }
            for j, obs in enumerate(obs_list):
                if len(obs) > MAX_OBSERVATION_LENGTH:
                    return {
                        "error": f"Entity {i}, observation {j} too long: {len(obs)} > {MAX_OBSERVATION_LENGTH} characters. Shorten or split the observation."
                    }

        results = []
        for entity_dict in entities:
            parsed = EntityInput.model_validate(entity_dict)

            # Upsert entity
            entity_id = store.upsert_entity(
                parsed.name, parsed.entityType, parsed.status
            )

            # Get existing observations
            existing_obs = store.get_observations(entity_id)
            existing_set = set(existing_obs)

            # Add new observations (skip duplicates)
            new_obs = [o for o in parsed.observations if o not in existing_set]
            if new_obs:
                store.add_observations(entity_id, new_obs)

            # Get final observations (existing + new)
            all_obs = store.get_observations(entity_id)

            # Recompute embedding
            _recompute_embedding(entity_id, parsed.name, parsed.entityType)

            # Limbic: initialize access tracking for new/updated entity
            store.init_access(entity_id)

            results.append(
                {
                    "name": parsed.name,
                    "entityType": parsed.entityType,
                    "observations": all_obs,
                }
            )

        return {"entities": results}
    except Exception as e:
        logger.error("Error in create_entities: %s", e)
        return {"error": str(e)}


# ============================================================
# TOOL: create_relations
# ============================================================
def create_relations(relations: list[dict[str, Any]]) -> dict[str, Any]:
    """Create relations between entities. Both entities must exist.
    Returns created relations or errors for missing entities."""
    try:
        store = _server_mod.store
        results = []
        errors = []

        for rel_dict in relations:
            parsed = RelationInput.model_validate(rel_dict)

            from_ent = store.get_entity_by_name(parsed.from_entity)
            to_ent = store.get_entity_by_name(parsed.to_entity)

            if not from_ent:
                errors.append(f"Entity not found: {parsed.from_entity}")
                results.append({"error": f"Entity not found: {parsed.from_entity}"})
                continue
            if not to_ent:
                errors.append(f"Entity not found: {parsed.to_entity}")
                results.append({"error": f"Entity not found: {parsed.to_entity}"})
                continue

            created = store.create_relation(
                from_ent["id"], to_ent["id"], parsed.relationType, parsed.context
            )
            if created:
                result_item = {
                    "from": parsed.from_entity,
                    "to": parsed.to_entity,
                    "relationType": parsed.relationType,
                }
                if parsed.context:
                    result_item["context"] = parsed.context
                results.append(result_item)
            else:
                results.append(
                    {
                        "from": parsed.from_entity,
                        "to": parsed.to_entity,
                        "relationType": parsed.relationType,
                        "error": "Relation already exists",
                    }
                )

        if errors:
            return {"relations": results, "errors": errors}
        return {"relations": results}
    except Exception as e:
        logger.error("Error in create_relations: %s", e)
        return {"error": str(e)}


# ============================================================
# TOOL: add_observations
# ============================================================
def add_observations(
    name: str,
    observations: list[str],
    kind: str = "generic",
    supersedes: int | None = None,
) -> dict[str, Any]:
    """Add observations to an existing entity.

    Args:
        name: Entity name.
        observations: List of observation strings to add.
        kind: The kind/type of observations (default 'generic').
        supersedes: Optional observation ID to supersede. The referenced obs
            will be marked as superseded, and the new obs will reference it."""
    try:
        store = _server_mod.store
        # Input validation
        if len(observations) > MAX_OBSERVATIONS_PER_CALL:
            return {
                "error": f"Too many observations: {len(observations)} > {MAX_OBSERVATIONS_PER_CALL}. Split into smaller batches."
            }
        for i, obs in enumerate(observations):
            if len(obs) > MAX_OBSERVATION_LENGTH:
                return {
                    "error": f"Observation {i} too long: {len(obs)} > {MAX_OBSERVATION_LENGTH} characters. Shorten or split the observation."
                }

        entity = store.get_entity_by_name(name)
        if not entity:
            return {"error": f"Entity not found: {name}"}

        store.add_observations(
            entity["id"], observations, kind=kind, supersedes=supersedes
        )
        all_obs = store.get_observations(entity["id"])

        # Recompute embedding
        _recompute_embedding(entity["id"], entity["name"], entity["entity_type"])

        # Limbic: record access on observation add
        store.record_access(entity["id"])

        return {
            "entity": {
                "name": entity["name"],
                "entityType": entity["entity_type"],
                "observations": all_obs,
            }
        }
    except Exception as e:
        logger.error("Error in add_observations: %s", e)
        return {"error": str(e)}


# ============================================================
# TOOL: delete_entities
# ============================================================
def delete_entities(entityNames: list[str]) -> dict[str, Any]:
    """Delete entities and all their relations/observations."""
    try:
        store = _server_mod.store
        if len(entityNames) > MAX_ENTITIES_PER_CALL:
            return {
                "error": f"Too many entity names: {len(entityNames)} > {MAX_ENTITIES_PER_CALL}. Delete fewer entities at a time."
            }

        deleted = []
        errors = []

        for name in entityNames:
            entity = store.get_entity_by_name(name)
            if not entity:
                errors.append(f"Entity not found: {name}")
                continue
            count = store.delete_entities_by_names([name])
            if count > 0:
                deleted.append(name)

        result: dict[str, Any] = {"deleted": deleted}
        if errors:
            result["errors"] = errors
        return result
    except Exception as e:
        logger.error("Error in delete_entities: %s", e)
        return {"error": str(e)}


# ============================================================
# TOOL: delete_observations
# ============================================================
def delete_observations(name: str, observations: list[str]) -> dict[str, Any]:
    """Delete specific observations from an entity."""
    try:
        store = _server_mod.store
        entity = store.get_entity_by_name(name)
        if not entity:
            return {"error": f"Entity not found: {name}"}

        store.delete_observations(entity["id"], observations)
        all_obs = store.get_observations(entity["id"])

        # Recompute embedding
        _recompute_embedding(entity["id"], entity["name"], entity["entity_type"])

        return {
            "entity": {
                "name": entity["name"],
                "entityType": entity["entity_type"],
                "observations": all_obs,
            }
        }
    except Exception as e:
        logger.error("Error in delete_observations: %s", e)
        return {"error": str(e)}


# ============================================================
# TOOL: delete_relations
# ============================================================
def delete_relations(relations: list[dict[str, Any]]) -> dict[str, Any]:
    """Delete relations between entities."""
    try:
        store = _server_mod.store
        deleted = []
        errors = []

        for rel_dict in relations:
            parsed = RelationInput.model_validate(rel_dict)

            from_ent = store.get_entity_by_name(parsed.from_entity)
            to_ent = store.get_entity_by_name(parsed.to_entity)

            if not from_ent or not to_ent:
                errors.append(
                    f"Entity not found: {parsed.from_entity} or {parsed.to_entity}"
                )
                continue

            was_deleted = store.delete_relation(
                from_ent["id"], to_ent["id"], parsed.relationType
            )
            if was_deleted:
                deleted.append(
                    {
                        "from": parsed.from_entity,
                        "to": parsed.to_entity,
                        "relationType": parsed.relationType,
                    }
                )
            else:
                errors.append(
                    f"Relation not found: {parsed.from_entity} -> "
                    f"{parsed.to_entity} ({parsed.relationType})"
                )

        result: dict[str, Any] = {"deleted": deleted}
        if errors:
            result["errors"] = errors
        return result
    except Exception as e:
        logger.error("Error in delete_relations: %s", e)
        return {"error": str(e)}
