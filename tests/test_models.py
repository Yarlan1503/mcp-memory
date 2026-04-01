import pytest
from pydantic import ValidationError
import sys

sys.path.insert(0, "src")
from mcp_memory.models import (
    EntityInput,
    EntityOutput,
    RelationInput,
    RelationOutput,
    LimbicScores,
    SearchResultItem,
    SearchEvent,
    SearchResultLog,
)


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


class TestEntityOutput:
    def test_entity_output(self):
        entity = EntityOutput(
            name="TestEntity", entityType="Person", observations=["obs1", "obs2"]
        )
        assert entity.name == "TestEntity"
        assert entity.entityType == "Person"
        assert entity.observations == ["obs1", "obs2"]


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


class TestLimbicScores:
    def test_limbic_scores(self):
        scores = LimbicScores(importance=0.8, temporal_factor=0.5, cooc_boost=1.2)
        assert scores.importance == 0.8
        assert scores.temporal_factor == 0.5
        assert scores.cooc_boost == 1.2


class TestSearchResultItem:
    def test_search_result_item(self):
        scores = LimbicScores(importance=0.9, temporal_factor=0.6, cooc_boost=1.1)
        item = SearchResultItem(
            name="Result1",
            entityType="Generic",
            observations=["test"],
            limbic_score=1.5,
            scoring=scores,
        )
        assert item.name == "Result1"
        assert item.entityType == "Generic"
        assert item.observations == ["test"]
        assert item.limbic_score == 1.5
        assert item.scoring == scores
        assert item.distance is None
        assert item.rrf_score is None

    def test_search_result_item_with_scores(self):
        scores = LimbicScores(importance=0.7, temporal_factor=0.3, cooc_boost=0.9)
        item = SearchResultItem(
            name="NestedScores",
            entityType="Person",
            observations=[],
            limbic_score=2.0,
            scoring=scores,
            distance=0.25,
            rrf_score=0.85,
        )
        assert item.scoring.importance == 0.7
        assert item.scoring.temporal_factor == 0.3
        assert item.scoring.cooc_boost == 0.9
        assert item.distance == 0.25
        assert item.rrf_score == 0.85


class TestSearchEvent:
    def test_search_event_defaults(self):
        event = SearchEvent(
            query_text="test query",
            treatment=1,
            k_limit=10,
            num_results=5,
            engine_used="limbic",
        )
        assert event.query_text == "test query"
        assert event.treatment == 1
        assert event.k_limit == 10
        assert event.num_results == 5
        assert event.duration_ms is None
        assert event.engine_used == "limbic"


class TestSearchResultLog:
    def test_search_result_log(self):
        log = SearchResultLog(
            entity_id=1,
            entity_name="Entity1",
            rank=1,
            limbic_score=1.5,
            cosine_sim=0.95,
            importance=0.8,
            temporal=0.6,
            cooc_boost=1.2,
            baseline_rank=2,
        )
        assert log.entity_id == 1
        assert log.entity_name == "Entity1"
        assert log.rank == 1
        assert log.limbic_score == 1.5
        assert log.cosine_sim == 0.95
        assert log.importance == 0.8
        assert log.temporal == 0.6
        assert log.cooc_boost == 1.2
        assert log.baseline_rank == 2
