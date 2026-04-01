"""Shared pytest fixtures for MCP Memory tests."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcp_memory.storage import MemoryStore


@pytest.fixture
def memory_store(tmp_path):
    """MemoryStore backed by temporary SQLite file."""
    db_path = str(tmp_path / "test_memory.db")
    store = MemoryStore(db_path)
    yield store
    store.close()


@pytest.fixture
def store_with_schema(memory_store):
    """MemoryStore with schema initialized."""
    memory_store.init_db()
    return memory_store


@pytest.fixture
def store_with_data(store_with_schema):
    """MemoryStore with sample entities and relations."""
    # Crear entidades de prueba
    e1 = store_with_schema.upsert_entity("TestEntity", "Testing")
    e2 = store_with_schema.upsert_entity("AnotherEntity", "Testing")
    store_with_schema.add_observations(e1, ["observation 1", "observation 2"])
    store_with_schema.add_observations(e2, ["another obs"])
    store_with_schema.create_relation(e1, e2, "related_to")
    return store_with_schema
