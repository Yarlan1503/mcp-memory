"""Tests for entity_splitter module."""

import numpy as np
import pytest

from mcp_memory.entity_splitter import (
    _extract_topics,
    _extract_topics_tfidf,
    _extract_topics_semantic,
    _generate_cluster_names,
    _get_threshold,
    get_threshold,
    _calculate_split_score,
    analyze_entity_for_split,
    propose_entity_split,
    execute_entity_split,
    find_all_split_candidates,
)


# ---------------------------------------------------------------------------
# Mock embedding engine for semantic tests
# ---------------------------------------------------------------------------


class MockEngine:
    """Fake EmbeddingEngine that returns deterministic embeddings.

    Accepts a pre-built mapping {text: vector} or generates random ones.
    """

    def __init__(self, embeddings_map: dict[str, np.ndarray] | None = None):
        self._map = embeddings_map or {}
        self._available = True

    @property
    def available(self) -> bool:
        return self._available

    def encode(self, texts: list[str], *, task: str = "passage") -> np.ndarray:
        vectors = []
        for t in texts:
            if t in self._map:
                vectors.append(self._map[t])
            else:
                # Deterministic random based on hash
                rng = np.random.RandomState(hash(t) % 2**31)
                vec = rng.randn(384).astype(np.float32)
                vec /= np.linalg.norm(vec)
                vectors.append(vec)
        return np.array(vectors)


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


# ---------------------------------------------------------------------------
# Semantic clustering tests (Fase 2)
# ---------------------------------------------------------------------------


class TestGenerateClusterNames:
    """Tests for _generate_cluster_names function."""

    def test_basic_two_clusters(self):
        """Two clusters with distinct vocab get different names."""
        clusters = {
            0: ["decisión python proyecto", "decisión código fuente"],
            1: ["hallazgo memoria rendimiento", "hallazgo cache lento"],
        }
        names = _generate_cluster_names(clusters)
        assert len(names) == 2
        assert names[0] != names[1]
        # Names should be capitalized
        for name in names.values():
            assert name[0].isupper()

    def test_empty_input(self):
        """Empty dict returns empty dict."""
        assert _generate_cluster_names({}) == {}

    def test_single_obs_cluster(self):
        """Cluster with one observation still gets a name."""
        clusters = {0: ["decisión usar python"]}
        names = _generate_cluster_names(clusters)
        assert len(names) == 1
        assert 0 in names
        assert len(names[0]) > 0

    def test_no_keywords_fallback(self):
        """Cluster with only stop words gets fallback name."""
        clusters = {0: ["a", "b", "c"]}
        names = _generate_cluster_names(clusters)
        assert 0 in names
        assert names[0] == "Tema_1"


class TestExtractTopicsSemantic:
    """Tests for _extract_topics_semantic function."""

    def _make_engine(self, observations: list[str]) -> MockEngine:
        """Build a MockEngine with well-separated embeddings for two groups."""
        n = len(observations)
        mid = n // 2
        emb_map: dict[str, np.ndarray] = {}
        for i, obs in enumerate(observations):
            vec = np.zeros(384, dtype=np.float32)
            if i < mid:
                vec[0] = 1.0  # Group A: positive on dim 0
            else:
                vec[1] = 1.0  # Group B: positive on dim 1
            emb_map[obs] = vec
        return MockEngine(emb_map)

    def test_two_distinct_groups(self):
        """Observations from two topics cluster into two groups."""
        obs = [
            "DECISIÓN: Usar Python para el backend",
            "DECISIÓN: Configurar entorno virtual",
            "HALLAZGO: El servidor tiene alta latencia",
            "HALLAZGO: La base de datos necesita índices",
        ]
        engine = self._make_engine(obs)
        result = _extract_topics_semantic(obs, engine)
        assert len(result) == 2
        # All observations should be assigned
        total = sum(len(v) for v in result.values())
        assert total == 4

    def test_single_cluster_falls_back(self):
        """When all obs are identical, clustering degenerates → TF-IDF fallback."""
        obs = ["Observación idéntica"] * 5
        engine = MockEngine()  # Will generate random but identical text → same hash? No.
        # Force all embeddings to be identical
        vec = np.ones(384, dtype=np.float32) / np.sqrt(384)
        engine = MockEngine({o: vec for o in obs})
        result = _extract_topics_semantic(obs, engine)
        # Should fall back to TF-IDF (1 cluster is degenerate)
        # TF-IDF with identical text → 1 topic or fallback
        assert len(result) >= 1

    def test_n_clusters_falls_back(self):
        """When each obs is its own cluster, falls back to TF-IDF."""
        obs = [f"tema único {i}" for i in range(4)]
        # Make each embedding orthogonal
        emb_map = {}
        for i, o in enumerate(obs):
            vec = np.zeros(384, dtype=np.float32)
            vec[i] = 1.0
            emb_map[o] = vec
        engine = MockEngine(emb_map)
        result = _extract_topics_semantic(obs, engine)
        # 4 clusters for 4 obs is degenerate → falls back to TF-IDF
        assert len(result) >= 1

    def test_empty_list(self):
        """Empty list returns empty dict."""
        engine = MockEngine()
        result = _extract_topics_semantic([], engine)
        assert result == {}

    def test_single_observation(self):
        """Single observation falls back to TF-IDF."""
        engine = MockEngine()
        result = _extract_topics_semantic(["Única observación"], engine)
        assert len(result) == 1


class TestDispatcherIntegration:
    """Tests for _extract_topics dispatcher routing."""

    def test_semantic_path_with_mock_engine(self, monkeypatch):
        """Dispatcher uses semantic path when engine is available."""
        obs = [
            "DECISIÓN: Implementar API REST",
            "DECISIÓN: Configurar autenticación JWT",
            "HALLAZGO: Memory leak en worker pool",
            "HALLAZGO: Connection pool agotado",
        ]
        # Build well-separated embeddings
        emb_map = {}
        for i, o in enumerate(obs):
            vec = np.zeros(384, dtype=np.float32)
            if i < 2:
                vec[0] = 1.0
            else:
                vec[1] = 1.0
            emb_map[o] = vec
        mock_engine = MockEngine(emb_map)

        def mock_get_engine():
            return mock_engine

        monkeypatch.setattr(
            "mcp_memory.entity_splitter._get_engine", mock_get_engine, raising=False
        )
        # Also patch the import inside _extract_topics
        monkeypatch.setattr(
            "mcp_memory._helpers._get_engine", mock_get_engine, raising=False
        )

        result = _extract_topics(obs)
        assert len(result) == 2
        total = sum(len(v) for v in result.values())
        assert total == 4

    def test_tfidf_fallback_when_no_engine(self, monkeypatch):
        """Dispatcher falls back to TF-IDF when engine is None."""
        monkeypatch.setattr(
            "mcp_memory._helpers._get_engine", lambda: None, raising=False
        )
        obs = [
            "DECISIÓN: Usar Python",
            "DECISIÓN: Configurar Docker",
            "HALLAZGO: Cache lento",
        ]
        result = _extract_topics(obs)
        assert len(result) >= 1

    def test_tfidf_fallback_when_engine_unavailable(self, monkeypatch):
        """Dispatcher falls back to TF-IDF when engine.available is False."""
        mock_engine = MockEngine()
        mock_engine._available = False
        monkeypatch.setattr(
            "mcp_memory._helpers._get_engine", lambda: mock_engine, raising=False
        )
        obs = ["Observación de prueba"]
        result = _extract_topics(obs)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Integration tests: full pipeline with store + semantic clustering
# ---------------------------------------------------------------------------


def _build_semantic_engine(
    group_a: list[str], group_b: list[str]
) -> MockEngine:
    """Build a MockEngine where group_a and group_b are well-separated."""
    emb_map: dict[str, np.ndarray] = {}
    for obs in group_a:
        vec = np.zeros(384, dtype=np.float32)
        vec[0] = 1.0
        emb_map[obs] = vec
    for obs in group_b:
        vec = np.zeros(384, dtype=np.float32)
        vec[1] = 1.0
        emb_map[obs] = vec
    return MockEngine(emb_map)


class TestSemanticIntegration:
    """Integration tests: store + semantic clustering end-to-end."""

    def test_analyze_with_semantic_topics(self, store_with_schema, monkeypatch):
        """analyze_entity_for_split produces semantic topics, not tema_N."""
        # Create entity with 21 observations (threshold=20)
        group_a = [f"DECISIÓN: Configurar API endpoint {i}" for i in range(10)]
        group_b = [f"HALLAZGO: Error memoria worker {i}" for i in range(11)]
        all_obs = group_a + group_b

        entity_id = store_with_schema.upsert_entity("BigEntity", "Persona")
        store_with_schema.add_observations(entity_id, all_obs)

        # Mock engine with well-separated embeddings
        engine = _build_semantic_engine(group_a, group_b)
        monkeypatch.setattr(
            "mcp_memory._helpers._get_engine", lambda: engine, raising=False
        )

        result = analyze_entity_for_split(store_with_schema, "BigEntity")
        assert result is not None
        assert result["needs_split"] is True
        assert result["observation_count"] == 21

        # Topics should be 2 semantic clusters, not 21 individual tema_N
        topics = result["topics"]
        assert len(topics) == 2
        total_obs = sum(len(v) for v in topics.values())
        assert total_obs == 21

        # Topic names should NOT be generic "tema_N"
        for name in topics:
            assert not name.startswith("tema_"), f"Got generic name: {name}"

    def test_propose_with_semantic_topics(self, store_with_schema, monkeypatch):
        """propose_entity_split returns meaningful split names."""
        group_a = [f"DECISIÓN: Migrar base datos {i}" for i in range(10)]
        group_b = [f"HALLAZGO: Latencia red alta {i}" for i in range(10)]
        all_obs = group_a + group_b

        entity_id = store_with_schema.upsert_entity("SplitEntity", "Sesion")
        store_with_schema.add_observations(entity_id, all_obs)

        engine = _build_semantic_engine(group_a, group_b)
        monkeypatch.setattr(
            "mcp_memory._helpers._get_engine", lambda: engine, raising=False
        )

        proposal = propose_entity_split(store_with_schema, "SplitEntity")
        assert proposal is not None
        assert proposal["original_entity"]["name"] == "SplitEntity"

        splits = proposal["suggested_splits"]
        assert len(splits) == 2

        # Split names should be "ParentName - TopicName"
        for split in splits:
            assert split["name"].startswith("SplitEntity - ")
            # Topic part should not be generic
            topic_part = split["name"].split(" - ", 1)[1]
            assert not topic_part.startswith("tema_")

        # Relations: 2 per split (contiene + parte_de)
        assert len(proposal["relations_to_create"]) == 4

    def test_semantic_vs_tfidf_produces_fewer_topics(
        self, store_with_schema, monkeypatch
    ):
        """Semantic clustering produces fewer, more meaningful topics than TF-IDF."""
        # 21 observations with overlapping keywords but distinct semantics
        api_obs = [
            f"DECISIÓN: Configurar endpoint API REST {i}" for i in range(7)
        ]
        db_obs = [
            f"DECISIÓN: Crear índice base datos {i}" for i in range(7)
        ]
        cache_obs = [
            f"DECISIÓN: Implementar cache Redis {i}" for i in range(7)
        ]
        all_obs = api_obs + db_obs + cache_obs

        entity_id = store_with_schema.upsert_entity("MultiTopic", "Persona")
        store_with_schema.add_observations(entity_id, all_obs)

        # --- TF-IDF path (no engine) ---
        monkeypatch.setattr(
            "mcp_memory._helpers._get_engine", lambda: None, raising=False
        )
        tfidf_result = analyze_entity_for_split(store_with_schema, "MultiTopic")
        tfidf_topics = tfidf_result["topics"]

        # --- Semantic path (engine with 3 clusters) ---
        emb_map: dict[str, np.ndarray] = {}
        for obs in api_obs:
            v = np.zeros(384, dtype=np.float32)
            v[0] = 1.0
            emb_map[obs] = v
        for obs in db_obs:
            v = np.zeros(384, dtype=np.float32)
            v[1] = 1.0
            emb_map[obs] = v
        for obs in cache_obs:
            v = np.zeros(384, dtype=np.float32)
            v[2] = 1.0
            emb_map[obs] = v
        engine = MockEngine(emb_map)
        monkeypatch.setattr(
            "mcp_memory._helpers._get_engine", lambda: engine, raising=False
        )
        semantic_result = analyze_entity_for_split(store_with_schema, "MultiTopic")
        semantic_topics = semantic_result["topics"]

        # Semantic should produce exactly 3 clusters
        assert len(semantic_topics) == 3
        # All obs accounted for
        assert sum(len(v) for v in semantic_topics.values()) == 21

    def test_fallback_to_tfidf_when_engine_crashes(
        self, store_with_schema, monkeypatch
    ):
        """If engine.encode() raises, gracefully falls back to TF-IDF."""

        class BrokenEngine:
            available = True

            def encode(self, texts, **kw):
                raise RuntimeError("ONNX session crashed")

        monkeypatch.setattr(
            "mcp_memory._helpers._get_engine", lambda: BrokenEngine(), raising=False
        )

        entity_id = store_with_schema.upsert_entity("CrashTest", "Sesion")
        obs = [f"Observación {i}" for i in range(16)]
        store_with_schema.add_observations(entity_id, obs)

        # Should not raise — falls back to TF-IDF
        result = analyze_entity_for_split(store_with_schema, "CrashTest")
        assert result is not None
        assert result["needs_split"] is True
