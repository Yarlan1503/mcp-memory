"""Tests for mcp_memory.config module."""

import sys

sys.path.insert(0, "src")
from mcp_memory import config


class TestConfigConstants:
    """Tests verifying configuration constants."""

    def test_max_observations_per_call(self):
        assert config.MAX_OBSERVATIONS_PER_CALL == 100

    def test_max_entities_per_call(self):
        assert config.MAX_ENTITIES_PER_CALL == 50

    def test_max_observation_length(self):
        assert config.MAX_OBSERVATION_LENGTH == 2000

    def test_max_query_length(self):
        assert config.MAX_QUERY_LENGTH == 500

    def test_use_ab_testing(self):
        assert config.USE_AB_TESTING is True

    def test_baseline_probability(self):
        assert config.BASELINE_PROBABILITY == 0.1
