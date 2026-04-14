from pydantic import BaseModel, Field


class EntityInput(BaseModel):
    """Input model for creating/updating entities."""

    name: str = Field(..., min_length=1)
    entityType: str = Field(default="Generic")
    observations: list[str] = Field(default_factory=list)
    status: str = Field(default="activo")


class RelationInput(BaseModel):
    """Input model for creating relations. Uses Anthropic-compatible aliases."""

    from_entity: str = Field(..., alias="from")
    to_entity: str = Field(..., alias="to")
    relationType: str
    context: str | None = None

    model_config = {"populate_by_name": True}
