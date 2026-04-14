import pytest
from pydantic import ValidationError

from mcp_memory.models import EntityInput, RelationInput


class TestEntityInput:
    def test_entity_input_valid(self):
        entity = EntityInput(name="Test")
        assert entity.name == "Test"
        assert entity.entityType == "Generic"
        assert entity.observations == []

    def test_entity_input_empty_name_fails(self):
        with pytest.raises(ValidationError):
            EntityInput(name="")

    def test_entity_input_custom_type(self):
        entity = EntityInput(name="Test", entityType="CustomType")
        assert entity.name == "Test"
        assert entity.entityType == "CustomType"
        assert entity.observations == []


class TestRelationInput:
    def test_relation_input_aliases(self):
        relation = RelationInput(
            from_entity="EntityA", to_entity="EntityB", relationType="relates_to"
        )
        assert relation.from_entity == "EntityA"
        assert relation.to_entity == "EntityB"
        assert relation.relationType == "relates_to"

    def test_relation_input_populate_by_name(self):
        relation = RelationInput(
            **{"from": "EntityA", "to": "EntityB", "relationType": "links_to"}
        )
        assert relation.from_entity == "EntityA"
        assert relation.to_entity == "EntityB"
        assert relation.relationType == "links_to"
