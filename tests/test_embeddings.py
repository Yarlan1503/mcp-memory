"""Tests for mcp_memory.embeddings module."""

import pytest
import numpy as np
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, "src")
from mcp_memory.embeddings import (
    EmbeddingEngine,
    serialize_f32,
    deserialize_f32,
    DIMENSION,
)


class TestPrepareEntityText:
    """Tests for prepare_entity_text static method."""

    def test_prepare_entity_text_no_observations(self):
        """Only name and type -> correct format."""
        result = EmbeddingEngine.prepare_entity_text("TestEntity", "Testing", [])
        assert result == "TestEntity (Testing)"

    def test_prepare_entity_text_with_observations(self):
        """Verifies observations are joined with ' | '."""
        result = EmbeddingEngine.prepare_entity_text("Test", "Type", ["obs1", "obs2"])
        assert "obs1" in result
        assert "obs2" in result
        assert " | " in result

    def test_prepare_entity_text_head_tail_selection(self):
        """With 15 obs, verifies head(3) + tail(7) + diverse middle selection."""
        observations = [f"obs{i}" for i in range(15)]
        result = EmbeddingEngine.prepare_entity_text("Entity", "Type", observations)
        # Should contain first 3 observations (head)
        assert "obs0" in result
        assert "obs1" in result
        assert "obs2" in result
        # Should contain last 7 observations (tail)
        assert "obs8" in result  # -7 means indices 8-14
        assert "obs14" in result
        # Total selected: 3 head + diverse middle + 7 tail
        # Header is "Entity (Type): "
        assert result.startswith("Entity (Type): ")

    def test_prepare_entity_text_with_relations(self):
        """Verifies relations are appended as 'Rel: ...'."""
        relations = [
            {"relation_type": "knows", "target_name": "Alice"},
            {"relation_type": "works_at", "target_name": "Acme"},
        ]
        result = EmbeddingEngine.prepare_entity_text(
            "Bob", "Person", ["obs1"], relations=relations
        )
        assert "Rel:" in result
        assert "knows" in result
        assert "Alice" in result
        assert "works_at" in result
        assert "Acme" in result


class TestSerializeF32:
    """Tests for serialize_f32 / deserialize_f32 binary helpers."""

    def test_serialize_f32(self):
        """Vector numpy -> bytes, verifies length = 384*4 = 1536 bytes."""
        vector = np.random.randn(DIMENSION).astype(np.float32)
        serialized = serialize_f32(vector)
        assert len(serialized) == DIMENSION * 4 == 1536

    def test_deserialize_f32(self):
        """Roundtrip: serialize -> deserialize -> values equal."""
        original = np.random.randn(DIMENSION).astype(np.float32)
        serialized = serialize_f32(original)
        deserialized = deserialize_f32(serialized)
        np.testing.assert_array_almost_equal(original, deserialized)


class TestEmbeddingEngineSingleton:
    """Tests for EmbeddingEngine singleton behavior."""

    def setup_method(self):
        """Reset singleton before each test."""
        EmbeddingEngine.reset()

    def teardown_method(self):
        """Reset singleton after each test."""
        EmbeddingEngine.reset()

    def test_embedding_engine_singleton(self):
        """Two calls to get_instance() return the same object."""
        engine1 = EmbeddingEngine.get_instance()
        engine2 = EmbeddingEngine.get_instance()
        assert engine1 is engine2

    def test_embedding_engine_reset(self):
        """reset() clears the singleton, next get_instance() creates new."""
        engine1 = EmbeddingEngine.get_instance()
        EmbeddingEngine.reset()
        engine2 = EmbeddingEngine.get_instance()
        assert engine1 is not engine2

    def test_embedding_engine_not_available_when_no_model(self):
        """When model files don't exist, available=False."""
        # Mock the model files as non-existent
        with patch("mcp_memory.embeddings.MODEL_DIR", "/nonexistent/path"):
            with patch("mcp_memory.embeddings.Path.exists", return_value=False):
                EmbeddingEngine.reset()
                engine = EmbeddingEngine.get_instance()
                assert engine.available is False
