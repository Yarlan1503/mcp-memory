"""Configuration constants for MCP Memory server."""

# ============================================================
# Input size limits
# ============================================================
MAX_OBSERVATIONS_PER_CALL = 100
MAX_ENTITIES_PER_CALL = 50
MAX_OBSERVATION_LENGTH = 2000
MAX_QUERY_LENGTH = 500

# --- A/B Testing Configuration ---
USE_AB_TESTING = True
BASELINE_PROBABILITY = 0.1  # 10% of queries are baseline (treatment=0)
