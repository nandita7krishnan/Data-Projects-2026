"""
LLM-based relationship extractor.

Given a free-text passage about AI, uses Ollama to extract
(source_concept, relationship_type, target_concept) triples
and ingests new knowledge into the graph.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from src.graph.schema import RELATIONSHIP_TYPES, ConceptNode, GraphClient

logger = logging.getLogger(__name__)

_EXTRACT_PROMPT = """\
You are a knowledge graph builder specializing in AI/ML concepts.

Extract all concept relationships from the text below.
Only use these relationship types: {rel_types}

Rules:
- Only extract relationships between AI/ML concepts
- Both source and target must be real AI/ML concepts
- Be conservative — only extract confident, clear relationships

Respond with ONLY valid JSON in this exact format:
{{
  "triples": [
    {{"source": "concept name", "relation": "RELATIONSHIP_TYPE", "target": "concept name"}},
    ...
  ]
}}

Text to analyze:
{text}
"""


def extract_relationships(
    text: str,
    ollama_client,
    model: str,
    known_concepts: Optional[set[str]] = None,
) -> list[dict]:
    """
    Use Ollama to extract (source, relation, target) triples from text.

    Returns a list of dicts: [{"source": ..., "relation": ..., "target": ...}]
    """
    prompt = _EXTRACT_PROMPT.format(
        rel_types=", ".join(RELATIONSHIP_TYPES),
        text=text[:4000],  # limit input length
    )

    try:
        response = ollama_client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},
        )
        content = response["message"]["content"]

        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if not json_match:
            logger.warning("No JSON found in extraction response.")
            return []

        parsed = json.loads(json_match.group())
        triples = parsed.get("triples", [])

        # Validate relation types
        valid = []
        for triple in triples:
            if triple.get("relation") in RELATIONSHIP_TYPES:
                valid.append(triple)
            else:
                logger.debug("Ignoring invalid relation type: %s", triple.get("relation"))

        return valid

    except Exception as e:
        logger.warning("Relationship extraction failed: %s", e)
        return []


def ingest_text(
    text: str,
    graph_client: GraphClient,
    ollama_client,
    model: str,
) -> dict:
    """
    Full ingestion pipeline for free text:
    1. Extract triples via LLM
    2. For each new concept name found, create a minimal stub node
    3. Add all relationships to the graph

    Returns stats: {triples_extracted, new_concepts, new_relationships}
    """
    known_names = set(graph_client.get_all_concept_names())
    triples = extract_relationships(text, ollama_client, model, known_names)

    new_concepts = 0
    new_relationships = 0

    for triple in triples:
        src = triple["source"]
        rel = triple["relation"]
        dst = triple["target"]

        # Auto-create stub nodes for unknown concepts
        for concept_name in [src, dst]:
            if concept_name not in known_names:
                stub = ConceptNode(
                    name=concept_name,
                    definition=f"Auto-extracted concept: {concept_name}",
                    category="Concept",
                    difficulty="intermediate",
                )
                graph_client.upsert_concept(stub)
                known_names.add(concept_name)
                new_concepts += 1
                logger.info("Created stub node: '%s'", concept_name)

        graph_client.add_relationship(src, rel, dst)
        new_relationships += 1

    return {
        "triples_extracted": len(triples),
        "new_concepts": new_concepts,
        "new_relationships": new_relationships,
    }
