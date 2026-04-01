"""Automatic entity splitting for MCP Memory.

Splits entities that exceed observation thresholds into focused sub-entities,
preserving the knowledge graph structure via contains/parte_de relations.

Conventions:
    - EntityTypes: Persona, Sistema, Componente, Proyecto, Diseno,
      ConocimientoTecnico, Tarea, Sesion
    - Thresholds: Sesion=15, Proyecto=25, Otras=20
    - Relations: contiene (parent→child), parte_de (child→parent)
    - Entity names preserve accents, entityType has no accent
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from storage import MemoryStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds by entity type
# ---------------------------------------------------------------------------
_THRESHOLDS: dict[str, int] = {
    "Sesion": 15,
    "Proyecto": 25,
}

_DEFAULT_THRESHOLD = 20

# ---------------------------------------------------------------------------
# Topic extraction constants
# ---------------------------------------------------------------------------
# Common Spanish stop words for keyword extraction
_STOP_WORDS: set[str] = {
    "de",
    "la",
    "que",
    "el",
    "en",
    "y",
    "a",
    "los",
    "del",
    "se",
    "las",
    "por",
    "un",
    "para",
    "con",
    "no",
    "una",
    "su",
    "al",
    "lo",
    "como",
    "más",
    "pero",
    "sus",
    "le",
    "ya",
    "o",
    "este",
    "sí",
    "porque",
    "esta",
    "entre",
    "cuando",
    "muy",
    "sin",
    "sobre",
    "también",
    "me",
    "hasta",
    "hay",
    "donde",
    "quien",
    "desde",
    "todo",
    "nos",
    "durante",
    "todos",
    "uno",
    "les",
    "ni",
    "contra",
    "otros",
    "ese",
    "eso",
    "ante",
    "ellos",
    "e",
    "esto",
    "mí",
    "antes",
    "algunos",
    "qué",
    "unos",
    "yo",
    "otro",
    "otras",
    "otra",
    "él",
    "tanto",
    "esa",
    "estos",
    "mucho",
    "quienes",
    "nada",
    "muchos",
    "cual",
    "poco",
    "ella",
    "estar",
    "estas",
    "algunas",
    "algo",
    "nosotros",
    "mi",
    "mis",
    "tú",
    "te",
    "ti",
    "tu",
    "tus",
    "ellas",
    "nosotras",
    "vosotros",
    "vosotras",
    "os",
    "mío",
    "mía",
    "míos",
    "mías",
    "tuyo",
    "tuya",
    "tuyos",
    "tuyas",
    "suyo",
    "suya",
    "suyos",
    "suyas",
    "nuestro",
    "nuestra",
    "nuestros",
    "nuestras",
    "vuestro",
    "vuestra",
    "vuestros",
    "vuestras",
    "esos",
    "esas",
    "estoy",
    "estás",
    "está",
    "estamos",
    "estáis",
    "están",
    "esté",
    "estés",
    "estemos",
    "estéis",
    "estén",
    "estaré",
    "estarás",
    "estará",
    "estaremos",
    "estaréis",
    "estarán",
    "estaría",
    "estarías",
    "estaríamos",
    "estaríais",
    "estarían",
    "estaba",
    "estabas",
    "estábamos",
    "estabais",
    "estaban",
    "estuve",
    "estuviste",
    "estuvo",
    "estuvimos",
    "estuvisteis",
    "estuvieron",
    "estuviera",
    "estuvieras",
    "estuviéramos",
    "estuvierais",
    "estuvieran",
    "estuviese",
    "estuvieses",
    "estuviésemos",
    "estuvieseis",
    "estuviesen",
    "estando",
    "estado",
    "estada",
    "estados",
    "estadas",
    "estad",
    "he",
    "has",
    "ha",
    "hemos",
    "habéis",
    "han",
    "haya",
    "hayas",
    "hayamos",
    "hayáis",
    "hayan",
    "habré",
    "habrás",
    "habrá",
    "habremos",
    "habréis",
    "habrán",
    "habría",
    "habrías",
    "habríamos",
    "habríais",
    "habrían",
    "había",
    "habías",
    "habíamos",
    "habíais",
    "habían",
    "hube",
    "hubiste",
    "hubo",
    "hubimos",
    "hubisteis",
    "hubieron",
    "hubiera",
    "hubieras",
    "hubiéramos",
    "hubierais",
    "hubieran",
    "hubiese",
    "hubieses",
    "hubiésemos",
    "hubieseis",
    "hubiesen",
    "habiendo",
    "habido",
    "habida",
    "habidos",
    "habidas",
    "soy",
    "eres",
    "es",
    "somos",
    "sois",
    "son",
    "sea",
    "seas",
    "seamos",
    "seáis",
    "sean",
    "seré",
    "serás",
    "será",
    "seremos",
    "seréis",
    "serán",
    "sería",
    "serías",
    "seríamos",
    "seríais",
    "serían",
    "era",
    "eras",
    "éramos",
    "erais",
    "eran",
    "fui",
    "fuiste",
    "fue",
    "fuimos",
    "fuisteis",
    "fueron",
    "fuera",
    "fueras",
    "fuéramos",
    "fuerais",
    "fueran",
    "fuese",
    "fueses",
    "fuésemos",
    "fueseis",
    "fuesen",
    "siendo",
    "sido",
    "tengo",
    "tienes",
    "tiene",
    "tenemos",
    "tenéis",
    "tienen",
    "tenga",
    "tengas",
    "tengamos",
    "tengáis",
    "tengan",
    "tendré",
    "tendrás",
    "tendrá",
    "tendremos",
    "tendréis",
    "tendrán",
    "tendría",
    "tendrías",
    "tendríamos",
    "tendríais",
    "tendrían",
    "tenía",
    "tenías",
    "teníamos",
    "teníais",
    "tenían",
    "tuve",
    "tuviste",
    "tuvo",
    "tuvimos",
    "tuvisteis",
    "tuvieron",
    "tuviera",
    "tuvieras",
    "tuviéramos",
    "tuvierais",
    "tuvieran",
    "tuviese",
    "tuvieses",
    "tuviésemos",
    "tuvieseis",
    "tuviesen",
    "teniendo",
    "tenido",
    "tenida",
    "tenidos",
    "tenidas",
    "tened",
}

# Minimum word length to consider
_MIN_WORD_LEN = 4

# Minimum document frequency for TF-IDF (as count)
_MIN_DOC_FREQ = 2


# ---------------------------------------------------------------------------
# Topic extraction
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Extract lowercase alphanumeric tokens from text."""
    tokens = re.findall(r"\b[a-záéíóúüñ]+\b", text.lower())
    return [t for t in tokens if len(t) >= _MIN_WORD_LEN and t not in _STOP_WORDS]


def _compute_tf(observations: list[str]) -> dict[str, float]:
    """Compute term frequency per document (observation)."""
    tf: dict[str, float] = Counter()
    for obs in observations:
        tokens = _tokenize(obs)
        for token in tokens:
            tf[token] += 1
    total = sum(tf.values())
    if total == 0:
        return {}
    return {term: count / total for term, count in tf.items()}


def _compute_idf(observations: list[str]) -> dict[str, float]:
    """Compute inverse document frequency across observations."""
    doc_count: dict[str, int] = Counter()
    for obs in observations:
        unique_tokens = set(_tokenize(obs))
        for token in unique_tokens:
            doc_count[token] += 1

    n_docs = len(observations)
    if n_docs == 0:
        return {}
    return {
        term: math.log(n_docs / df)
        for term, df in doc_count.items()
        if df >= _MIN_DOC_FREQ
    }


def _extract_topics(observations: list[str]) -> dict[str, list[str]]:
    """Extract topics from observations using TF-IDF and grouping.

    Returns a dict mapping topic_name -> [list_of_observations].
    Each topic represents a cluster of observations sharing common keywords.
    """
    if not observations:
        return {}

    n_obs = len(observations)

    # Compute TF and IDF
    tf = _compute_tf(observations)
    idf = _compute_idf(observations)

    if not tf or not idf:
        # Fallback: treat each observation as its own topic if no keywords found
        return {f"tema_{i + 1}": [obs] for i, obs in enumerate(observations)}

    # Compute TF-IDF scores for each term
    tfidf_scores: dict[str, float] = {}
    for term, tf_val in tf.items():
        if term in idf:
            tfidf_scores[term] = tf_val * idf[term]

    if not tfidf_scores:
        return {f"tema_{i + 1}": [obs] for i, obs in enumerate(observations)}

    # Get top keywords as topic anchors
    sorted_terms = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
    top_keywords = [term for term, _ in sorted_terms[:10]]

    # Assign each observation to a topic based on keyword overlap
    topic_assignments: dict[str, list[tuple[float, str]]] = {}

    for i, obs in enumerate(observations):
        tokens = set(_tokenize(obs))
        obs_tfidf = sum(tfidf_scores.get(t, 0) for t in tokens)

        # Find the best matching topic anchor
        best_topic = None
        best_score = -1.0

        for keyword in top_keywords:
            if keyword in tokens:
                # Score by cumulative TF-IDF of matching keywords
                score = sum(tfidf_scores.get(t, 0) for t in tokens if t == keyword)
                if score > best_score:
                    best_score = score
                    best_topic = keyword

        if best_topic is None:
            # No keyword match → create a generic topic
            best_topic = f"tema_{i + 1}"
        else:
            # Normalize topic name (capitalize first letter)
            best_topic = best_topic.capitalize()

        if best_topic not in topic_assignments:
            topic_assignments[best_topic] = []
        topic_assignments[best_topic].append((obs_tfidf, obs))

    # Sort observations within each topic by TF-IDF score
    result: dict[str, list[str]] = {}
    for topic, scored_obs in topic_assignments.items():
        scored_obs.sort(key=lambda x: x[0], reverse=True)
        result[topic] = [obs for _, obs in scored_obs]

    # If we have too few topics (merging happened), ensure minimum topic separation
    if len(result) < 2 and n_obs >= 4:
        # Force split into 2 roughly equal topics
        mid = n_obs // 2
        result = {
            "Tema_A": observations[:mid],
            "Tema_B": observations[mid:],
        }

    return result


# ---------------------------------------------------------------------------
# Split candidate detection
# ---------------------------------------------------------------------------


def _get_threshold(entity_type: str) -> int:
    """Return observation threshold for entity type."""
    return _THRESHOLDS.get(entity_type, _DEFAULT_THRESHOLD)


def _calculate_split_score(obs_count: int, threshold: int, num_topics: int) -> float:
    """Calculate split score combining count overflow and topic diversity.

    Score > 1.0 means candidate for splitting.
    """
    count_score = obs_count / threshold if threshold > 0 else 1.0
    # Reward topic diversity but cap contribution
    topic_bonus = min(num_topics / max(obs_count, 1), 0.5)
    return count_score + topic_bonus


def analyze_entity_for_split(
    store: MemoryStore, entity_name: str
) -> dict[str, Any] | None:
    """Analyze an entity to determine if it needs splitting.

    Args:
        store: MemoryStore instance
        entity_name: Name of the entity to analyze

    Returns:
        Dict with analysis results or None if entity not found.
        Keys: entity_name, entity_type, observation_count, threshold,
              needs_split, topics, split_score, suggested_splits
    """
    entity = store.get_entity_by_name(entity_name)
    if entity is None:
        logger.warning("Entity '%s' not found for split analysis", entity_name)
        return None

    entity_id = entity["id"]
    observations = store.get_observations(entity_id)
    obs_count = len(observations)
    entity_type = entity["entity_type"]

    threshold = _get_threshold(entity_type)

    # Extract topics
    topics = _extract_topics(observations)
    num_topics = len(topics)

    # Calculate split score
    split_score = _calculate_split_score(obs_count, threshold, num_topics)
    needs_split = split_score > 1.0 and obs_count > threshold

    logger.debug(
        "Analyzed entity '%s' (type=%s): obs=%d, threshold=%d, "
        "topics=%d, score=%.2f, needs_split=%s",
        entity_name,
        entity_type,
        obs_count,
        threshold,
        num_topics,
        split_score,
        needs_split,
    )

    return {
        "entity_name": entity_name,
        "entity_type": entity_type,
        "observation_count": obs_count,
        "threshold": threshold,
        "needs_split": needs_split,
        "topics": topics,
        "split_score": split_score,
    }


# ---------------------------------------------------------------------------
# Split proposal
# ---------------------------------------------------------------------------


def propose_entity_split(store: MemoryStore, entity_name: str) -> dict[str, Any] | None:
    """Analyze and propose a split for an entity if needed.

    Args:
        store: MemoryStore instance
        entity_name: Name of the entity to analyze

    Returns:
        Split proposal dict with keys:
            - original_entity: entity info (name, type, id)
            - suggested_splits: list of {name, entity_type, observations}
            - relations_to_create: list of {from, to, type}
        Returns None if entity not found or doesn't need splitting.
    """
    analysis = analyze_entity_for_split(store, entity_name)
    if analysis is None:
        return None

    if not analysis["needs_split"]:
        logger.info(
            "Entity '%s' does not need splitting (score=%.2f, obs=%d, threshold=%d)",
            entity_name,
            analysis["split_score"],
            analysis["observation_count"],
            analysis["threshold"],
        )
        return None

    topics = analysis["topics"]
    original_name = analysis["entity_name"]
    entity_type = analysis["entity_type"]

    # Build suggested splits from topics
    suggested_splits: list[dict[str, Any]] = []
    for topic_name, topic_obs in topics.items():
        suggested_splits.append(
            {
                "name": f"{original_name} - {topic_name}",
                "entity_type": entity_type,
                "observations": topic_obs,
            }
        )

    # Build relations (parent contains child, child is part of parent)
    relations_to_create: list[dict[str, str]] = []
    for split in suggested_splits:
        relations_to_create.extend(
            [
                {"from": original_name, "to": split["name"], "type": "contiene"},
                {"from": split["name"], "to": original_name, "type": "parte_de"},
            ]
        )

    proposal = {
        "original_entity": {
            "name": original_name,
            "entity_type": entity_type,
        },
        "suggested_splits": suggested_splits,
        "relations_to_create": relations_to_create,
        "analysis": {
            "observation_count": analysis["observation_count"],
            "threshold": analysis["threshold"],
            "split_score": analysis["split_score"],
            "num_topics": len(topics),
        },
    }

    logger.info(
        "Proposed split for '%s': %d new entities from %d topics",
        original_name,
        len(suggested_splits),
        len(topics),
    )

    return proposal


# ---------------------------------------------------------------------------
# Split execution
# ---------------------------------------------------------------------------


def execute_entity_split(
    store: MemoryStore,
    entity_name: str,
    approved_splits: list[dict[str, Any]],
    parent_entity_name: str | None = None,
) -> dict[str, Any]:
    """Execute an approved entity split.

    Args:
        store: MemoryStore instance
        entity_name: Name of the original entity to split
        approved_splits: List of approved split definitions.
            Each dict must have: name, entity_type, observations
        parent_entity_name: Optional explicit parent name (defaults to entity_name)

    Returns:
        Dict with execution results:
            - new_entities: list of created entity names
            - moved_observations: count of observations moved
            - relations_created: count of relations created
            - original_observations_remaining: observations left in original entity
    """
    parent = parent_entity_name or entity_name

    # Verify original entity exists
    original = store.get_entity_by_name(entity_name)
    if original is None:
        raise ValueError(f"Original entity '{entity_name}' not found")

    original_id = original["id"]
    original_type = original["entity_type"]

    # Collect all observations to move
    all_moved_obs: set[str] = set()
    for split in approved_splits:
        all_moved_obs.update(split["observations"])

    logger.info(
        "Executing split for '%s': %d new entities, %d observations to move",
        entity_name,
        len(approved_splits),
        len(all_moved_obs),
    )

    new_entity_ids: list[int] = []
    observations_moved = 0

    with store.db:
        # Create new entities and add observations
        for split in approved_splits:
            split_name = split["name"]
            split_type = split.get("entity_type", original_type)
            split_obs = split["observations"]

            # Upsert the new entity
            new_id = store.upsert_entity(split_name, split_type)
            new_entity_ids.append(new_id)

            # Add observations to new entity
            if split_obs:
                inserted = store.add_observations(new_id, split_obs)
                observations_moved += inserted
                logger.debug(
                    "Created entity '%s' (id=%d) with %d observations",
                    split_name,
                    new_id,
                    inserted,
                )

            # Create relations: parent contiene child, child parte_de parent
            store.create_relation(original_id, new_id, "contiene")
            store.create_relation(new_id, original_id, "parte_de")
            logger.debug(
                "Created contains/parte_de relations: %s <-> %s",
                parent,
                split_name,
            )

        # Remove moved observations from original entity
        if all_moved_obs:
            deleted = store.delete_observations(original_id, list(all_moved_obs))
            logger.debug(
                "Removed %d observations from original entity '%s'",
                deleted,
                entity_name,
            )

    # Get remaining observations in original entity
    remaining_obs = store.get_observations(original_id)

    result = {
        "new_entities": [s["name"] for s in approved_splits],
        "moved_observations": observations_moved,
        "relations_created": len(approved_splits) * 2,  # 2 per split
        "original_observations_remaining": len(remaining_obs),
    }

    logger.info(
        "Split executed successfully: %s created, %d obs moved, %d relations",
        result["new_entities"],
        observations_moved,
        result["relations_created"],
    )

    return result


# ---------------------------------------------------------------------------
# Mass analysis
# ---------------------------------------------------------------------------


def find_all_split_candidates(store: MemoryStore) -> list[dict[str, Any]]:
    """Find all entities in the store that are candidates for splitting.

    Args:
        store: MemoryStore instance

    Returns:
        List of analysis dicts for each candidate entity.
        Empty list if no candidates found.
    """
    entities = store.get_all_entities()
    candidates: list[dict[str, Any]] = []

    for entity in entities:
        analysis = analyze_entity_for_split(store, entity["name"])
        if analysis is not None and analysis["needs_split"]:
            candidates.append(analysis)
            logger.debug(
                "Split candidate: '%s' (score=%.2f, obs=%d/%d)",
                entity["name"],
                analysis["split_score"],
                analysis["observation_count"],
                analysis["threshold"],
            )

    logger.info(
        "Found %d split candidates out of %d total entities",
        len(candidates),
        len(entities),
    )

    return candidates
