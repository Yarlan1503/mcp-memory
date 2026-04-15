"""Tests for MCP Memory v3 integration: status, kind, supersedes, relations v3, reflections."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcp_memory.storage import MemoryStore


@pytest.fixture
def store(tmp_path):
    """MemoryStore with schema initialized."""
    db_path = str(tmp_path / "test_v3_integration.db")
    s = MemoryStore(db_path)
    s.init_db()
    yield s
    s.close()


# ==============================================================
# 1. prepare_entity_text with new fields
# ==============================================================


class TestPrepareEntityText:
    def test_prepare_entity_text_includes_status(self):
        """Status appears in the embedding text header."""
        from mcp_memory.embeddings import EmbeddingEngine

        text = EmbeddingEngine.prepare_entity_text(
            name="TestEnt",
            entity_type="Proyecto",
            observations=["obs1"],
            status="completado",
        )
        assert "[completado]" in text
        assert "TestEnt (Proyecto) [completado]:" in text

    def test_prepare_entity_text_includes_kind(self):
        """Non-generic kinds appear as [kind] prefix in observations."""
        from mcp_memory.embeddings import EmbeddingEngine

        obs = [
            {"content": "MORENA poder 96.2%", "kind": "hallazgo"},
            {"content": "generic observation", "kind": "generic"},
            {"content": "FECHA: 2026-04-09", "kind": "metadata"},
        ]
        text = EmbeddingEngine.prepare_entity_text(
            name="TestEnt",
            entity_type="Proyecto",
            observations=obs,
            status="activo",
        )
        assert "[hallazgo] MORENA poder 96.2%" in text
        assert "[metadata] FECHA: 2026-04-09" in text
        # Generic obs should NOT have [generic] prefix
        assert "[generic]" not in text
        # But the plain text should be there
        assert "generic observation" in text


# ==============================================================
# 2. Integration full v3
# ==============================================================


class TestFullV3Flow:
    def test_full_v3_flow(self, store, monkeypatch):
        """End-to-end: create entity with status -> add obs with kind -> supersede
        -> create relation with context -> add reflection -> open_nodes verifies all."""
        import mcp_memory.server as server_module
        from mcp_memory.server import open_nodes, add_reflection

        monkeypatch.setattr(server_module, "store", store)

        # 1. Create entities with status
        e1 = store.upsert_entity("Proyecto Test", "Proyecto", status="activo")
        e2 = store.upsert_entity("Sistema Test", "Sistema", status="completado")

        # 2. Add observations with kind
        store.add_observations(e1, ["Observación hallazgo"], kind="hallazgo")
        store.add_observations(e1, ["Estado inicial del proyecto"], kind="estado")
        store.add_observations(e1, ["Metadata del proyecto"], kind="metadata")

        # 3. Supersede an observation
        obs_data = store.get_observations_with_ids(e1)
        estado_obs = next(o for o in obs_data if o["kind"] == "estado")
        store.add_observations(
            e1,
            ["Estado actualizado del proyecto"],
            kind="estado",
            supersedes=estado_obs["id"],
        )

        # 4. Create relation with context
        store.create_relation(
            e1, e2, "producido_por", context="proyecto genera sistema"
        )

        # 5. Add reflection
        add_reflection("entity", e1, "nolan", "Este proyecto fue clave", "satisfaccion")

        # 6. open_nodes and verify everything
        result = open_nodes(["Proyecto Test"])
        assert "error" not in result
        entities = result["entities"]
        assert len(entities) == 1

        ent = entities[0]
        assert ent["name"] == "Proyecto Test"
        assert ent["entityType"] == "Proyecto"
        assert ent["status"] == "activo"

        # Observations: hallazgo + metadata + updated estado (old estado superseded)
        obs_kinds = {o["kind"] for o in ent["observations"]}
        assert "hallazgo" in obs_kinds
        assert "metadata" in obs_kinds
        assert "estado" in obs_kinds

        # The superseded obs should NOT appear
        obs_contents = [o["content"] for o in ent["observations"]]
        assert "Estado inicial del proyecto" not in obs_contents
        assert "Estado actualizado del proyecto" in obs_contents

        # Relations should include the target
        assert "relations" in ent
        rel_targets = {r["target_name"] for r in ent["relations"]}
        assert "Sistema Test" in rel_targets

        # Reflections should be present
        assert "reflections" in ent
        assert len(ent["reflections"]) == 1
        assert ent["reflections"][0]["content"] == "Este proyecto fue clave"

    def test_search_semantic_with_v3_entities(self, store, monkeypatch):
        """search_semantic works correctly with status, kind, and supersedes."""
        from mcp_memory.server import search_semantic
        from mcp_memory.scoring import RoutingStrategy

        monkeypatch.setattr("mcp_memory.server.store", store)

        # Setup some entities
        e1 = store.upsert_entity("SearchTest1", "Proyecto", status="activo")
        e2 = store.upsert_entity("SearchTest2", "Proyecto", status="archivado")
        store.add_observations(e1, ["active obs"], kind="hallazgo")
        store.add_observations(e2, ["archived obs"], kind="generic")

        with patch("mcp_memory.tools.search._get_engine") as mock_engine_fn:
            mock_engine = MagicMock()
            mock_engine.available = True
            mock_engine_fn.return_value = mock_engine

            # Mock store methods that search_semantic calls
            store.search_embeddings = MagicMock(
                return_value=[
                    {"entity_id": e1, "distance": 0.3},
                    {"entity_id": e2, "distance": 0.4},
                ]
            )
            store.search_fts = MagicMock(return_value=[])
            store.get_access_data = MagicMock(
                return_value={
                    e1: {"access_count": 5, "last_access": "2025-01-01"},
                    e2: {"access_count": 5, "last_access": "2025-01-01"},
                }
            )
            store.get_entity_degrees = MagicMock(return_value={e1: 3, e2: 3})
            store.get_co_occurrences = MagicMock(return_value={})
            store.get_access_days = MagicMock(return_value={e1: 10, e2: 10})
            store.get_entity_by_id = MagicMock(
                side_effect=lambda eid: {
                    "id": eid,
                    "name": "SearchTest1" if eid == e1 else "SearchTest2",
                    "entity_type": "Proyecto",
                    "status": "activo" if eid == e1 else "archivado",
                    "created_at": "2024-01-01",
                }
            )
            store.get_observations = MagicMock(return_value=["obs1"])
            store.get_observations_with_ids = MagicMock(
                return_value=[
                    {
                        "id": 1,
                        "content": "obs1",
                        "similarity_flag": 0,
                        "kind": "generic",
                        "supersedes": None,
                        "superseded_at": None,
                    }
                ]
            )
            store.log_search_event = MagicMock(return_value=1)

            with (
                patch("mcp_memory.tools.search._get_treatment", return_value=1),
                patch("mcp_memory.scoring.rank_candidates") as mock_rank,
                patch("mcp_memory.scoring.detect_query_type") as mock_detect,
            ):
                mock_detect.return_value = RoutingStrategy.HYBRID_BALANCED
                mock_rank.return_value = [
                    {
                        "entity_id": e1,
                        "limbic_score": 0.9,
                        "importance": 0.5,
                        "temporal_factor": 1.0,
                        "cooc_boost": 0.0,
                    },
                    {
                        "entity_id": e2,
                        "limbic_score": 0.9,
                        "importance": 0.5,
                        "temporal_factor": 1.0,
                        "cooc_boost": 0.0,
                    },
                ]

                result = search_semantic("test query", limit=5)

                # Should return results without crashing
                assert "results" in result
                assert len(result["results"]) == 2
                # activo should rank higher than archivado (0.5 multiplier)
                scores = {r["name"]: r["limbic_score"] for r in result["results"]}
                assert scores[f"SearchTest1"] > scores[f"SearchTest2"]
                # archivado score: 0.9 * 0.5 = 0.45
                assert scores[f"SearchTest2"] == pytest.approx(0.45, abs=0.01)

    def test_search_reflections_does_not_contaminate_entity_search(
        self, store, monkeypatch
    ):
        """Adding reflections does not affect entity search results."""
        import mcp_memory.server as server_module

        monkeypatch.setattr(server_module, "store", store)

        # Create entity and add reflection with unique content
        e1 = store.upsert_entity("ReflectOnlyEntity", "Testing")
        store.add_reflection("entity", e1, "nolan", "Unica reflexion unica xyz123")

        # search_entities should NOT find reflection content
        results = store.search_entities("xyz123")
        # search_entities only searches entity names, types, and observation content
        entity_names = [r["name"] for r in results]
        assert "ReflectOnlyEntity" not in entity_names

        # The entity should be findable by name
        results_by_name = store.search_entities("ReflectOnlyEntity")
        assert len(results_by_name) == 1
        # But reflection content should NOT be in observations
        assert all(
            "xyz123" not in o
            for r in results_by_name
            for o in r.get("observations", [])
        )

    def test_inverse_relations_appear_in_open_nodes(self, store, monkeypatch):
        """When creating contiene relation, inverse parte_de appears in open_nodes for both entities."""
        import mcp_memory.server as server_module
        from mcp_memory.server import open_nodes

        monkeypatch.setattr(server_module, "store", store)

        e1 = store.upsert_entity("Parent", "Sistema")
        e2 = store.upsert_entity("Child", "Componente")
        store.create_relation(e1, e2, "contiene", context="parent contains child")

        # Open parent — should see contiene -> Child
        result = open_nodes(["Parent"])
        rels = result["entities"][0]["relations"]
        rel_types = {
            (r["relation_type"], r["target_name"], r["direction"]) for r in rels
        }
        assert ("contiene", "Child", "from") in rel_types

        # Open child — should see parte_de -> Parent (auto-inverse)
        result = open_nodes(["Child"])
        rels = result["entities"][0]["relations"]
        rel_types = {
            (r["relation_type"], r["target_name"], r["direction"]) for r in rels
        }
        assert ("parte_de", "Parent", "from") in rel_types


# ==============================================================
# 3. Entity type migration script tests
# ==============================================================


class TestMigrationScript:
    @pytest.fixture(autouse=True)
    def _import_migration(self):
        """Import the migration module from scripts/ via importlib."""
        import importlib.util

        script_path = (
            Path(__file__).resolve().parents[1] / "scripts" / "migrate_entity_types.py"
        )
        spec = importlib.util.spec_from_file_location(
            "migrate_entity_types", script_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.migrate_entity_types = mod.migrate_entity_types

    def test_migration_script_migrates_types(self, store):
        """Script migrates entity types correctly."""
        # Create entities with old types
        store.upsert_entity("Sub Entity", "Subproyecto")
        store.upsert_entity("Mejora Entity", "Mejora")
        store.upsert_entity("Keep Entity", "Persona")

        # Run migration
        self.migrate_entity_types(store)

        # Verify types changed
        e1 = store.get_entity_by_name("Sub Entity")
        assert e1["entity_type"] == "Proyecto"
        e2 = store.get_entity_by_name("Mejora Entity")
        assert e2["entity_type"] == "Proyecto"
        e3 = store.get_entity_by_name("Keep Entity")
        assert e3["entity_type"] == "Persona"

    def test_migration_script_migrates_names_with_prefix(self, store):
        """Script renames entity names with old type prefix."""
        store.upsert_entity("Subproyecto: BD Congreso", "Subproyecto")
        store.upsert_entity("Mejora: Fix Bug", "Mejora")
        store.upsert_entity("NoPrefix", "Subproyecto")

        # Run migration
        self.migrate_entity_types(store)

        # Verify names changed where applicable
        assert store.get_entity_by_name("Proyecto: BD Congreso") is not None
        assert store.get_entity_by_name("Proyecto: Fix Bug") is not None
        assert (
            store.get_entity_by_name("NoPrefix") is not None
        )  # No prefix, name unchanged


# ==============================================================
# 4. Deterministic search_semantic with treatment mock
# ==============================================================


class TestDeterministicSearch:
    def test_search_semantic_deterministic_with_treatment_mock(
        self, store, monkeypatch
    ):
        """Verify that mocking _get_treatment makes search_semantic deterministic."""
        from mcp_memory.server import search_semantic
        from mcp_memory.scoring import RoutingStrategy

        monkeypatch.setattr("mcp_memory.server.store", store)

        with patch("mcp_memory.tools.search._get_engine") as mock_engine_fn:
            mock_engine = MagicMock()
            mock_engine.available = True
            mock_engine_fn.return_value = mock_engine

            store.upsert_entity("DeterministicEnt", "Test", status="activo")

            with (
                patch("mcp_memory.tools.search._get_treatment", return_value=1),
                patch("mcp_memory.scoring.reciprocal_rank_fusion"),
                patch("mcp_memory.scoring.rank_candidates") as mock_rank,
                patch("mcp_memory.scoring.detect_query_type") as mock_detect,
            ):
                mock_detect.return_value = RoutingStrategy.HYBRID_BALANCED
                mock_rank.return_value = [
                    {
                        "entity_id": 1,
                        "limbic_score": 0.95,
                        "importance": 0.5,
                        "temporal_factor": 1.0,
                        "cooc_boost": 0.0,
                    }
                ]

                # Setup store mocks that search_semantic needs
                store.search_embeddings = MagicMock(
                    return_value=[{"entity_id": 1, "distance": 0.3}]
                )
                store.search_fts = MagicMock(return_value=[])
                store.get_access_data = MagicMock(
                    return_value={1: {"access_count": 5, "last_access": "2025-01-01"}}
                )
                store.get_entity_degrees = MagicMock(return_value={1: 3})
                store.get_co_occurrences = MagicMock(return_value={})
                store.get_access_days = MagicMock(return_value={1: 10})
                store.get_entity_by_id = MagicMock(
                    return_value={
                        "id": 1,
                        "name": "DeterministicEnt",
                        "entity_type": "Test",
                        "status": "activo",
                        "created_at": "2024-01-01",
                    }
                )
                store.get_observations = MagicMock(return_value=["obs1"])
                store.get_observations_with_ids = MagicMock(
                    return_value=[
                        {
                            "id": 1,
                            "content": "obs1",
                            "similarity_flag": 0,
                            "kind": "generic",
                            "supersedes": None,
                            "superseded_at": None,
                        }
                    ]
                )
                store.log_search_event = MagicMock(return_value=1)

                result = search_semantic("test query", limit=5)

                # Should always have limbic_score when treatment=1 is forced
                assert "results" in result
                if result["results"]:
                    assert "limbic_score" in result["results"][0]
                    assert "routing_strategy" in result["results"][0]
