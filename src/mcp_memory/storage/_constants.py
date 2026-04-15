"""Constants for the MCP Memory storage layer."""

# --- Phase 3: Inverse relation map ---
INVERSE_RELATIONS: dict[str, str] = {
    "contiene": "parte_de",
    "parte_de": "contiene",
}

# --- Phase 3: Legacy relation type normalization ---
LEGACY_RELATION_TYPES: dict[str, tuple[str, str]] = {
    "continua": ("contribuye_a", "sesión continuación"),
    "documentado_en": ("producido_por", "documentado en"),
}

# --- Phase 4: Reflections validation constants ---
VALID_TARGET_TYPES = {"entity", "session", "relation", "global"}
VALID_AUTHORS = {"nolan", "sofia"}
VALID_MOODS = {"frustracion", "satisfaccion", "curiosidad", "duda", "insight"}
