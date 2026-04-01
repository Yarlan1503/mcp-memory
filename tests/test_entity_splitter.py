"""Tests for entity_splitter module."""

import pytest

from mcp_memory.entity_splitter import (
    _extract_topics,
    _get_threshold,
    _calculate_split_score,
    analyze_entity_for_split,
    propose_entity_split,
    execute_entity_split,
    find_all_split_candidates,
)


class TestExtractTopics:
    """Tests for _extract_topics function."""

    def test_extract_topics_with_decision_prefix(self):
        """Test extraction with DECISIÓN prefix observations."""
        observations = [
            "DECISIÓN: Usar Python para el proyecto",
            "DECISIÓN: Implementar cache TTL de 5 minutos",
            "DECISIÓN: Mover a base de datos SQLite",
        ]
        topics = _extract_topics(observations)
        # Should group all under one topic (Decisión)
        assert len(topics) >= 1

    def test_extract_topics_with_hallazgo_prefix(self):
        """Test extraction with HALLAZGO prefix observations."""
        observations = [
            "HALLAZGO: El sistema usa mucha memoria",
            "HALLAZGO: El cache mejora rendimiento 3x",
            "HALLAZGO: La base de datos tiene índices faltantes",
        ]
        topics = _extract_topics(observations)
        # Should group all under one topic (Hallazgo)
        assert len(topics) >= 1

    def test_extract_topics_mixed_prefixes(self):
        """Test extraction with mixed prefix observations."""
        observations = [
            "DECISIÓN: Usar Python para el proyecto",
            "HALLAZGO: El sistema usa mucha memoria",
            "DECISIÓN: Implementar cache TTL",
            "HALLAZGO: El cache mejora rendimiento",
        ]
        topics = _extract_topics(observations)
        # Should separate decisions from hallazgos
        assert len(topics) >= 2

    def test_extract_topics_no_keywords_fallback(self):
        """Test fallback when observations have no keywords."""
        # Short words that get filtered out by stop words and min length
        observations = ["a", "b", "c", "d"]
        topics = _extract_topics(observations)
        # Should fall back to individual topics
        assert len(topics) == len(observations)

    def test_extract_topics_empty_list(self):
        """Test with empty observations list."""
        topics = _extract_topics([])
        assert topics == {}

    def test_extract_topics_single_observation(self):
        """Test with single observation."""
        observations = ["DECISIÓN: Implementar autenticación JWT"]
        topics = _extract_topics(observations)
        assert len(topics) == 1


class TestAnalyzeEntityForSplit:
    """Tests for analyze_entity_for_split function."""

    def test_entity_not_found(self, store_with_schema):
        """Test analysis of non-existent entity returns None."""
        result = analyze_entity_for_split(store_with_schema, "NonExistent")
        assert result is None

    def test_entity_does_not_need_split(self, store_with_schema):
        """Test entity with few observations doesn't need split."""
        entity_id = store_with_schema.upsert_entity("SmallEntity", "Persona")
        store_with_schema.add_observations(entity_id, ["obs 1", "obs 2"])
        result = analyze_entity_for_split(store_with_schema, "SmallEntity")
        assert result is not None
        assert result["needs_split"] is False
        assert result["observation_count"] == 2

    def test_entity_needs_split_sesion_over_threshold(self, store_with_schema):
        """Test Sesion entity with >15 observations needs split."""
        entity_id = store_with_schema.upsert_entity("TestSession", "Sesion")
        # Add 16 observations to trigger split for Sesion (threshold=15)
        observations = [
            f"Observación número {i} del proyecto python" for i in range(16)
        ]
        store_with_schema.add_observations(entity_id, observations)
        result = analyze_entity_for_split(store_with_schema, "TestSession")
        assert result is not None
        assert result["needs_split"] is True
        assert result["observation_count"] == 16
        assert result["threshold"] == 15
        assert result["entity_type"] == "Sesion"

    def test_entity_needs_split_regular_threshold(self, store_with_schema):
        """Test entity with >20 observations needs split (default threshold)."""
        entity_id = store_with_schema.upsert_entity("LargeEntity", "Persona")
        # Add 21 observations to trigger split (default threshold=20)
        observations = [f"Observación {i} con contenido específico" for i in range(21)]
        store_with_schema.add_observations(entity_id, observations)
        result = analyze_entity_for_split(store_with_schema, "LargeEntity")
        assert result is not None
        assert result["needs_split"] is True
        assert result["observation_count"] == 21


class TestProposeEntitySplit:
    """Tests for propose_entity_split function."""

    def test_propose_no_split_needed(self, store_with_schema):
        """Test propose returns None when entity doesn't need split."""
        entity_id = store_with_schema.upsert_entity("SmallEntity", "Componente")
        store_with_schema.add_observations(entity_id, ["obs 1", "obs 2"])
        result = propose_entity_split(store_with_schema, "SmallEntity")
        assert result is None

    def test_propose_returns_valid_split_proposal(self, store_with_schema):
        """Test propose returns valid split proposal when needed."""
        entity_id = store_with_schema.upsert_entity("SplitSession", "Sesion")
        # Add 16 observations to trigger split for Sesion
        observations = [f"DECISIÓN: Item de decisión {i}" for i in range(10)] + [
            f"HALLAZGO: Item de hallazgo {i}" for i in range(6)
        ]
        store_with_schema.add_observations(entity_id, observations)
        result = propose_entity_split(store_with_schema, "SplitSession")
        assert result is not None
        assert "original_entity" in result
        assert "suggested_splits" in result
        assert "relations_to_create" in result
        assert result["original_entity"]["name"] == "SplitSession"
        assert len(result["suggested_splits"]) >= 1
        # Check relations structure
        assert len(result["relations_to_create"]) >= 2

    def test_propose_entity_not_found(self, store_with_schema):
        """Test propose returns None for non-existent entity."""
        result = propose_entity_split(store_with_schema, "NonExistent")
        assert result is None


class TestExecuteEntitySplit:
    """Tests for execute_entity_split function."""

    def test_execute_split_creates_new_entities(self, store_with_schema):
        """Test execute creates new entities from split."""
        entity_id = store_with_schema.upsert_entity("SplitTest", "Sesion")
        observations = [f"DECISIÓN: Decisión {i}" for i in range(10)] + [
            f"HALLAZGO: Hallazgo {i}" for i in range(6)
        ]
        store_with_schema.add_observations(entity_id, observations)

        # First get the proposal
        proposal = propose_entity_split(store_with_schema, "SplitTest")
        assert proposal is not None

        # Execute the split
        result = execute_entity_split(
            store_with_schema,
            "SplitTest",
            proposal["suggested_splits"],
        )

        assert result["moved_observations"] > 0
        assert len(result["new_entities"]) == len(proposal["suggested_splits"])
        assert result["relations_created"] == len(proposal["suggested_splits"]) * 2

    def test_execute_split_removes_moved_observations(self, store_with_schema):
        """Test execute removes moved observations from original."""
        entity_id = store_with_schema.upsert_entity("MoveTest", "Sesion")
        observations = [f"Observación {i}" for i in range(16)]
        store_with_schema.add_observations(entity_id, observations)

        proposal = propose_entity_split(store_with_schema, "MoveTest")
        assert proposal is not None

        result = execute_entity_split(
            store_with_schema,
            "MoveTest",
            proposal["suggested_splits"],
        )

        # Original should have fewer observations
        assert result["original_observations_remaining"] < 16

    def test_execute_split_nonexistent_entity_raises(self, store_with_schema):
        """Test execute raises error for non-existent entity."""
        with pytest.raises(ValueError, match="not found"):
            execute_entity_split(
                store_with_schema,
                "NonExistent",
                [{"name": "New", "entity_type": "Sesion", "observations": []}],
            )

    def test_execute_split_creates_relations(self, store_with_schema):
        """Test execute creates contiene/parte_de relations."""
        entity_id = store_with_schema.upsert_entity("RelTest", "Sesion")
        observations = [f"Obs {i}" for i in range(16)]
        store_with_schema.add_observations(entity_id, observations)

        proposal = propose_entity_split(store_with_schema, "RelTest")
        assert proposal is not None

        result = execute_entity_split(
            store_with_schema,
            "RelTest",
            proposal["suggested_splits"],
        )

        # Relations created = 2 per split (contiene + parte_de)
        assert result["relations_created"] == len(proposal["suggested_splits"]) * 2


class TestFindAllSplitCandidates:
    """Tests for find_all_split_candidates function."""

    def test_find_candidates_empty_db(self, store_with_schema):
        """Test find returns empty list for empty database."""
        candidates = find_all_split_candidates(store_with_schema)
        assert candidates == []

    def test_find_candidates_finds_large_entity(self, store_with_schema):
        """Test find correctly identifies entities needing split."""
        # Create entity that needs split
        entity_id = store_with_schema.upsert_entity("LargeSession", "Sesion")
        observations = [f"Observación {i} con contenido python" for i in range(16)]
        store_with_schema.add_observations(entity_id, observations)

        # Create entity that doesn't need split
        small_id = store_with_schema.upsert_entity("SmallSession", "Sesion")
        store_with_schema.add_observations(small_id, ["obs 1", "obs 2"])

        candidates = find_all_split_candidates(store_with_schema)

        assert len(candidates) == 1
        assert candidates[0]["entity_name"] == "LargeSession"
        assert candidates[0]["needs_split"] is True

    def test_find_candidates_multiple(self, store_with_schema):
        """Test find returns multiple candidates."""
        # Create two entities needing split
        for i in range(2):
            eid = store_with_schema.upsert_entity(f"Large{i}", "Sesion")
            obs = [f"Observación {j} del proyecto" for j in range(16)]
            store_with_schema.add_observations(eid, obs)

        candidates = find_all_split_candidates(store_with_schema)
        assert len(candidates) == 2


class TestThresholds:
    """Tests for threshold constants and functions."""

    def test_get_threshold_sesion(self):
        """Test Sesion threshold is 15."""
        assert _get_threshold("Sesion") == 15

    def test_get_threshold_proyecto(self):
        """Test Proyecto threshold is 25."""
        assert _get_threshold("Proyecto") == 25

    def test_get_threshold_default(self):
        """Test default threshold is 20."""
        assert _get_threshold("UnknownType") == 20

    def test_calculate_split_score_needs_split(self):
        """Test split score > 1 indicates needs split."""
        # 16 obs / 15 threshold = 1.07 + bonus
        score = _calculate_split_score(16, 15, 2)
        assert score > 1.0

    def test_calculate_split_score_no_split(self):
        """Test split score <= 1 indicates no split needed."""
        # 10 obs / 15 threshold = 0.67
        score = _calculate_split_score(10, 15, 2)
        assert score <= 1.0
