from pydantic import BaseModel, Field


class EntityInput(BaseModel):
    """Input model for creating/updating entities."""

    name: str = Field(..., min_length=1)
    entityType: str = Field(default="Generic")
    observations: list[str] = Field(default_factory=list)
    status: str = Field(default="activo")


class EntityOutput(BaseModel):
    """Output model for entities."""

    name: str
    entityType: str
    observations: list[str] | list[dict]  # Can be list[dict] with kind metadata


class RelationInput(BaseModel):
    """Input model for creating relations. Uses Anthropic-compatible aliases."""

    from_entity: str = Field(..., alias="from")
    to_entity: str = Field(..., alias="to")
    relationType: str
    context: str | None = None

    model_config = {"populate_by_name": True}


class RelationOutput(BaseModel):
    """Output model for relations."""

    from_entity: str = Field(..., alias="from")
    to_entity: str = Field(..., alias="to")
    relationType: str

    model_config = {"populate_by_name": True}


# === A/B Testing Models ===


class LimbicScores(BaseModel):
    """Desglose de componentes del scoring límbico."""

    importance: float
    temporal_factor: float
    cooc_boost: float


class SearchResultItem(BaseModel):
    """Un resultado individual de search_semantic con metadata de scoring."""

    name: str
    entityType: str
    observations: list[str]
    limbic_score: float
    scoring: LimbicScores
    distance: float | None = None
    rrf_score: float | None = None


class BaselineResult(BaseModel):
    """Resultado de ranking baseline (cosine-only)."""

    entity_id: int
    entity_name: str
    cosine_sim: float
    rank: int


class SearchEvent(BaseModel):
    """Evento de búsqueda para logging."""

    query_text: str
    treatment: int  # 1=limbic, 0=baseline
    k_limit: int
    num_results: int
    duration_ms: float | None = None
    engine_used: str


class SearchResultLog(BaseModel):
    """Un resultado individual para logging."""

    entity_id: int
    entity_name: str
    rank: int
    limbic_score: float
    cosine_sim: float
    importance: float
    temporal: float
    cooc_boost: float
    baseline_rank: int | None = None
