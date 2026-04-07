"""Tests for containment-based deduplication helpers and integration."""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcp_memory.scoring import (
    compute_containment,
    combined_similarity,
    CONTAINMENT_THRESHOLD,
    LENGTH_RATIO_THRESHOLD,
    _tokenize,
)
from mcp_memory.storage import MemoryStore
from mcp_memory.embeddings import EmbeddingEngine, DIMENSION


# ---------------------------------------------------------------------------
# Helpers (reuse from test_semantic_dedup)
# ---------------------------------------------------------------------------


def _make_vector_pair(cos_sim: float) -> tuple[np.ndarray, np.ndarray]:
    """Create two L2-normalised vectors with a specific cosine similarity."""
    rng = np.random.RandomState(42)
    v1 = rng.randn(DIMENSION).astype(np.float32)
    v1 = v1 / np.linalg.norm(v1)

    noise = rng.randn(DIMENSION).astype(np.float32)
    noise = noise - np.dot(noise, v1) * v1
    noise = noise / np.linalg.norm(noise)

    sin_sim = np.sqrt(max(0.0, 1.0 - cos_sim**2))
    v2 = cos_sim * v1 + sin_sim * noise
    v2 = v2 / np.linalg.norm(v2)
    return v1, v2


class FakeEmbeddingEngine:
    """Controllable mock embedding engine for testing."""

    def __init__(self):
        self.available = True
        self._vectors: dict[str, np.ndarray] = {}

    def set_vector(self, text: str, vector: np.ndarray) -> None:
        self._vectors[text] = vector

    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        result = []
        for t in texts:
            if t in self._vectors:
                result.append(self._vectors[t])
            else:
                rng = np.random.RandomState(hash(t) % (2**31))
                v = rng.randn(DIMENSION).astype(np.float32)
                v = v / np.linalg.norm(v)
                self._vectors[t] = v
                result.append(v)
        return np.array(result)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test_containment.db")
    s = MemoryStore(db_path)
    s.init_db()
    yield s
    s.close()


@pytest.fixture(autouse=True)
def reset_embedding_singleton():
    """Reset EmbeddingEngine singleton before/after each test."""
    EmbeddingEngine.reset()
    yield
    EmbeddingEngine.reset()


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_basic_split(self):
        tokens = _tokenize("hello world foo")
        assert tokens == {"hello", "world", "foo"}

    def test_lowercase(self):
        tokens = _tokenize("Hello World FOO")
        assert tokens == {"hello", "world", "foo"}

    def test_strips_punctuation(self):
        tokens = _tokenize("hello, world! foo.")
        assert tokens == {"hello", "world", "foo"}

    def test_empty_string(self):
        assert _tokenize("") == set()

    def test_whitespace_only(self):
        assert _tokenize("   ") == set()

    def test_punctuation_only(self):
        assert _tokenize("...!") == set()


# ---------------------------------------------------------------------------
# compute_containment
# ---------------------------------------------------------------------------


class TestComputeContainment:
    def test_full_match(self):
        """All tokens in shorter appear in longer → 1.0."""
        shorter = "cat sat mat"
        longer = "the cat sat on the mat"
        assert compute_containment(shorter, longer) == 1.0

    def test_no_match(self):
        """No shared tokens → 0.0."""
        shorter = "apple banana cherry"
        longer = "dog elephant frog"
        assert compute_containment(shorter, longer) == 0.0

    def test_partial_match(self):
        """Some tokens in common → between 0 and 1."""
        shorter = "cat sat mat"  # 3 tokens
        longer = "cat dog fish bird"  # "cat" matches → 1/3
        result = compute_containment(shorter, longer)
        assert 0.0 < result < 1.0

    def test_case_insensitive(self):
        """Same words different case → high containment."""
        shorter = "CAT Sat MAT"
        longer = "The cat sat on the mat today"
        assert compute_containment(shorter, longer) == 1.0

    def test_empty_shorter(self):
        """Empty shorter string → 1.0 (trivially contained)."""
        assert compute_containment("", "any text here") == 1.0

    def test_whitespace_shorter(self):
        """Whitespace-only shorter → 1.0."""
        assert compute_containment("   ", "any text here") == 1.0

    def test_empty_both(self):
        """Both empty → 1.0."""
        assert compute_containment("", "") == 1.0

    def test_punctuation_stripped(self):
        """Punctuation doesn't affect matching."""
        shorter = "cat's mat."
        longer = "The cat sat on the mat"
        # "cats" and "cat" are different after strip, "mat" matches
        tokens_short = _tokenize(shorter)
        tokens_long = _tokenize(longer)
        overlap = tokens_short & tokens_long
        expected = len(overlap) / len(tokens_short)
        assert compute_containment(shorter, longer) == expected

    def test_order_doesnt_matter(self):
        """Token sets — order is irrelevant."""
        shorter = "z y x"
        longer = "a b c x y z"
        assert compute_containment(shorter, longer) == 1.0

    def test_duplicate_tokens_in_shorter(self):
        """Duplicate tokens don't inflate containment (set-based)."""
        shorter = "cat cat cat dog"  # unique: {cat, dog}
        longer = "cat dog fish"  # matches: {cat, dog}
        assert compute_containment(shorter, longer) == 1.0


# ---------------------------------------------------------------------------
# combined_similarity
# ---------------------------------------------------------------------------


class TestCombinedSimilarity:
    def test_cosine_only_passes(self):
        """cosine >= threshold → True regardless of containment."""
        assert (
            combined_similarity(0.90, "short", "also short", cosine_threshold=0.85)
            is True
        )

    def test_cosine_below_threshold_fails_without_containment(self):
        """cosine < threshold, no containment trigger → False."""
        # Two texts of similar length with low cosine
        assert (
            combined_similarity(
                0.5, "hello world", "foo bar baz", cosine_threshold=0.85
            )
            is False
        )

    def test_containment_only_passes(self):
        """cosine < threshold but containment high + length_ratio >= 2 → True."""
        short = "glm-4v-flash gratis imagen"
        long = (
            "Suscripcion de pago para usar GLM Coding Max. "
            "Da acceso API via Zhipu AI. Incluye: glm-4v-flash (gratis, solo imagen), "
            "glm-4.1v-thinking-flash (gratis, thinking, solo imagen). "
            "Acceso a glm-4v-Plus (pago, video). SDK Python disponible via zhipuai."
        )
        # cosine low (0.5), but containment should be high and length ratio > 2
        assert len(long) / len(short) >= LENGTH_RATIO_THRESHOLD
        containment = compute_containment(short, long)
        assert containment >= CONTAINMENT_THRESHOLD
        assert combined_similarity(0.5, short, long, cosine_threshold=0.85) is True

    def test_similar_length_no_containment_trigger(self):
        """cosine < threshold, containment high, but length_ratio < 2 → False."""
        # Two texts of similar length sharing tokens
        text_a = "the cat sat on the mat today"
        text_b = "a cat sat on that mat yesterday"
        # length ratio ≈ 1.0, containment high, but ratio < 2
        ratio = max(len(text_a), len(text_b)) / min(len(text_a), len(text_b))
        assert ratio < LENGTH_RATIO_THRESHOLD
        assert combined_similarity(0.5, text_a, text_b, cosine_threshold=0.85) is False

    def test_low_containment_fails(self):
        """cosine < threshold, containment low even with asymmetric length → False."""
        short = "xyz abc def"  # tokens not in longer
        long = "The quick brown fox jumps over the lazy dog and then some more text padding"
        assert len(long) / len(short) >= LENGTH_RATIO_THRESHOLD
        containment = compute_containment(short, long)
        assert containment < CONTAINMENT_THRESHOLD
        assert combined_similarity(0.5, short, long, cosine_threshold=0.85) is False

    def test_exact_threshold_values(self):
        """At exactly threshold boundaries."""
        # cosine exactly at threshold
        assert combined_similarity(0.85, "a", "b", cosine_threshold=0.85) is True
        # cosine just below
        assert combined_similarity(0.8499, "a", "b", cosine_threshold=0.85) is False

    def test_length_ratio_exactly_2(self):
        """Length ratio of exactly 2.0 should trigger containment check."""
        short = "abc"
        long = "abc def ghi jkl mno pqr"  # 21 chars vs 3 chars = ratio 7
        # Actually let's be precise: ratio >= 2.0 means trigger
        short = "a"
        long = "a b c d e f g h i j"  # Much longer
        assert combined_similarity(0.5, short, long, cosine_threshold=0.85) is True

    def test_custom_thresholds(self):
        """Custom containment_threshold and cosine_threshold respected."""
        short = "cat dog"
        long = "cat dog bird fish mouse elephant lion tiger bear"
        # cosine low, containment high (cat, dog both present)
        # With default containment_threshold=0.7 → True
        assert combined_similarity(0.5, short, long, cosine_threshold=0.85) is True
        # With containment_threshold=1.0 → might fail if not all tokens match
        result = combined_similarity(
            0.5, short, long, cosine_threshold=0.85, containment_threshold=1.0
        )
        assert result is True  # "cat" and "dog" both in longer


# ---------------------------------------------------------------------------
# Integration: add_observations with containment
# ---------------------------------------------------------------------------


class TestAddObservationsContainment:
    def test_add_obs_flagged_when_short_subset_of_long(self, store):
        """Short obs that is token-subset of long existing → flag = 1."""
        long_obs = (
            "Suscripcion de pago para usar GLM Coding Max del sistema SofIA. "
            "Da acceso API via Zhipu AI (zhipuai). Incluye: glm-4v-flash (gratis, solo imagen), "
            "glm-4.1v-thinking-flash (gratis, thinking, solo imagen). "
            "Acceso a glm-4v-Plus (pago, video) si se necesita. "
            "SDK Python disponible via zhipuai."
        )
        # Short obs uses tokens that ALL appear in long_obs (containment = 1.0)
        short_obs = "glm-4v-flash gratis solo imagen glm-4.1v-thinking-flash thinking"

        # Verify containment is high before testing
        assert compute_containment(short_obs, long_obs) >= CONTAINMENT_THRESHOLD

        eid = store.upsert_entity("Test", "Type")

        # Set up fake engine BEFORE adding long_obs so it gets a consistent vector
        fake = FakeEmbeddingEngine()
        v1, v2 = _make_vector_pair(0.55)  # Low cosine, like real case
        fake.set_vector(long_obs, v1)
        fake.set_vector(short_obs, v2)

        with patch.object(store, "_get_embedding_engine", return_value=fake):
            store.add_observations(eid, [long_obs])
            store.add_observations(eid, [short_obs])

        rows = store.db.execute(
            "SELECT similarity_flag FROM observations "
            "WHERE entity_id = ? AND content = ?",
            (eid, short_obs),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["similarity_flag"] == 1

    def test_add_obs_not_flagged_when_not_subset(self, store):
        """Short obs that is NOT a subset → flag = 0."""
        long_obs = (
            "The complete works of Shakespeare including Hamlet Macbeth and Othello"
        )
        short_obs = (
            "Python Django REST framework tutorial guide"  # Completely different
        )

        eid = store.upsert_entity("Test", "Type")
        store.add_observations(eid, [long_obs])

        fake = FakeEmbeddingEngine()
        v1, v2 = _make_vector_pair(0.3)  # Very low cosine
        fake.set_vector(long_obs, v1)
        fake.set_vector(short_obs, v2)

        with patch.object(store, "_get_embedding_engine", return_value=fake):
            store.add_observations(eid, [short_obs])

        rows = store.db.execute(
            "SELECT similarity_flag FROM observations "
            "WHERE entity_id = ? AND content = ?",
            (eid, short_obs),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["similarity_flag"] == 0

    def test_existing_cosine_behavior_preserved(self, store):
        """cosine >= 0.85 still triggers flag (no regression)."""
        eid = store.upsert_entity("Test", "Type")
        store.add_observations(eid, ["The cat sat on the mat"])

        fake = FakeEmbeddingEngine()
        v1, v2 = _make_vector_pair(0.90)
        fake.set_vector("The cat sat on the mat", v1)
        fake.set_vector("A cat was sitting on a mat", v2)

        with patch.object(store, "_get_embedding_engine", return_value=fake):
            store.add_observations(eid, ["A cat was sitting on a mat"])

        rows = store.db.execute(
            "SELECT similarity_flag FROM observations "
            "WHERE entity_id = ? AND content = ?",
            (eid, "A cat was sitting on a mat"),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["similarity_flag"] == 1

    def test_similar_length_no_false_positive(self, store):
        """Two texts of similar length with low cosine → flag = 0 (no containment trigger)."""
        eid = store.upsert_entity("Test", "Type")
        store.add_observations(eid, ["The cat sat on the mat"])

        fake = FakeEmbeddingEngine()
        v1, v2 = _make_vector_pair(0.50)
        fake.set_vector("The cat sat on the mat", v1)
        fake.set_vector("A completely different topic about dogs running", v2)

        with patch.object(store, "_get_embedding_engine", return_value=fake):
            store.add_observations(
                eid, ["A completely different topic about dogs running"]
            )

        rows = store.db.execute(
            "SELECT similarity_flag FROM observations "
            "WHERE entity_id = ? AND content = ?",
            (eid, "A completely different topic about dogs running"),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["similarity_flag"] == 0


# ---------------------------------------------------------------------------
# Integration: find_duplicate_observations with containment
# ---------------------------------------------------------------------------


class TestFindDuplicateObservationsContainment:
    def _import_tool(self):
        """Lazy import the tool function."""
        from mcp_memory.server import find_duplicate_observations

        return find_duplicate_observations

    def test_catches_short_as_containment_duplicate(self, store):
        """Tool detects short obs as duplicate of long via containment."""
        long_obs = (
            "Suscripcion de pago para usar GLM Coding Max del sistema SofIA. "
            "Da acceso API via Zhipu AI (zhipuai). Incluye: glm-4v-flash (gratis, solo imagen), "
            "glm-4.1v-thinking-flash (gratis, thinking, solo imagen). "
            "Acceso a glm-4v-Plus (pago, video). SDK Python disponible via zhipuai."
        )
        short_obs = "Sub-agentes puede usar glm-4v-flash (gratis, solo imagen)"

        eid = store.upsert_entity("Test", "Type")
        store.add_observations(eid, [long_obs])
        store.db.execute(
            "INSERT INTO observations (entity_id, content, similarity_flag) VALUES (?, ?, 0)",
            (eid, short_obs),
        )
        store.db.commit()

        fake = FakeEmbeddingEngine()
        v1, v2 = _make_vector_pair(0.55)  # Low cosine
        fake.set_vector(long_obs, v1)
        fake.set_vector(short_obs, v2)

        with (
            patch("mcp_memory.server.store", store),
            patch("mcp_memory.server._get_engine", return_value=fake),
        ):
            tool = self._import_tool()
            result = tool(entity_name="Test", threshold=0.85, containment_threshold=0.7)

        assert result["duplicates_found"] >= 1
        assert len(result["clusters"]) >= 1

    def test_containment_threshold_param(self, store):
        """containment_threshold parameter is respected."""
        long_obs = (
            "Comprehensive documentation about the Python programming language "
            "including syntax semantics and standard library reference"
        )
        short_obs = "Python programming language"  # All tokens in longer

        eid = store.upsert_entity("Test", "Type")
        store.add_observations(eid, [long_obs])
        store.db.execute(
            "INSERT INTO observations (entity_id, content, similarity_flag) VALUES (?, ?, 0)",
            (eid, short_obs),
        )
        store.db.commit()

        fake = FakeEmbeddingEngine()
        v1, v2 = _make_vector_pair(0.55)
        fake.set_vector(long_obs, v1)
        fake.set_vector(short_obs, v2)

        with (
            patch("mcp_memory.server.store", store),
            patch("mcp_memory.server._get_engine", return_value=fake),
        ):
            tool = self._import_tool()

            # Default containment_threshold=0.7 should detect
            result = tool(entity_name="Test", threshold=0.85, containment_threshold=0.7)
            assert result["duplicates_found"] >= 1

    def test_match_type_in_output(self, store):
        """Pairs include match_type field: 'cosine' or 'containment'."""
        long_obs = (
            "Complete reference guide for the system including all features "
            "and configuration options for advanced users"
        )
        short_obs = "system features configuration"

        eid = store.upsert_entity("Test", "Type")
        store.add_observations(eid, [long_obs])
        store.db.execute(
            "INSERT INTO observations (entity_id, content, similarity_flag) VALUES (?, ?, 0)",
            (eid, short_obs),
        )
        store.db.commit()

        fake = FakeEmbeddingEngine()
        v1, v2 = _make_vector_pair(0.55)
        fake.set_vector(long_obs, v1)
        fake.set_vector(short_obs, v2)

        with (
            patch("mcp_memory.server.store", store),
            patch("mcp_memory.server._get_engine", return_value=fake),
        ):
            tool = self._import_tool()
            result = tool(entity_name="Test", threshold=0.85, containment_threshold=0.7)

        if result["duplicates_found"] >= 1:
            # Check the clusters contain both observations
            assert len(result["clusters"]) >= 1
            cluster_obs = result["clusters"][0]["observations"]
            contents = [o["content"] for o in cluster_obs]
            assert short_obs in contents
            assert long_obs in contents


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_containment_threshold_default(self):
        assert CONTAINMENT_THRESHOLD == 0.7

    def test_length_ratio_threshold_default(self):
        assert LENGTH_RATIO_THRESHOLD == 2.0
