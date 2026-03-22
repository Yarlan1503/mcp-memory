from pydantic import BaseModel, Field


class EntityInput(BaseModel):
    """Input model for creating/updating entities."""

    name: str = Field(..., min_length=1)
    entityType: str = Field(default="Generic")
    observations: list[str] = Field(default_factory=list)


class EntityOutput(BaseModel):
    """Output model for entities."""

    name: str
    entityType: str
    observations: list[str]


class RelationInput(BaseModel):
    """Input model for creating relations. Uses Anthropic-compatible aliases."""

    from_entity: str = Field(..., alias="from")
    to_entity: str = Field(..., alias="to")
    relationType: str

    model_config = {"populate_by_name": True}


class RelationOutput(BaseModel):
    """Output model for relations."""

    from_entity: str = Field(..., alias="from")
    to_entity: str = Field(..., alias="to")
    relationType: str

    model_config = {"populate_by_name": True}
