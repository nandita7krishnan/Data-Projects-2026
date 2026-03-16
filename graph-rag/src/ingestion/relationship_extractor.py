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
from pathlib import Path
from typing import Optional

from config.settings import PROCESSED_CHUNKS_PATH
from src.graph.schema import RELATIONSHIP_TYPES, ConceptNode, GraphClient, TextChunk

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


def batch_ingest_corpus(
    chunks: list[TextChunk],
    graph_client: GraphClient,
    ollama_client,
    model: str,
    refresh_known_every: int = 20,
) -> dict:
    """
    Run relationship extraction over every TextChunk and accumulate results
    into the graph. Tracks processed chunk_ids in a sidecar JSON file so
    the process can be resumed if interrupted.

    Args:
        chunks: list of TextChunk objects from document_loader
        graph_client: the graph backend to write into
        ollama_client: connected Ollama client
        model: Ollama model name
        refresh_known_every: refresh known concept names from graph every N chunks

    Returns:
        stats dict: {chunks_processed, chunks_skipped, triples_total,
                     new_concepts_total, new_relationships_total}
    """
    # Load already-processed chunk IDs
    processed_ids: set[str] = set()
    if PROCESSED_CHUNKS_PATH.exists():
        try:
            with open(PROCESSED_CHUNKS_PATH, encoding="utf-8") as f:
                processed_ids = set(json.load(f))
        except Exception:
            processed_ids = set()

    chunks_processed = 0
    chunks_skipped = 0
    triples_total = 0
    new_concepts_total = 0
    new_relationships_total = 0

    known_names: set[str] = set(graph_client.get_all_concept_names())
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        if chunk.chunk_id in processed_ids:
            chunks_skipped += 1
            continue

        print(f"  [{i+1}/{total}] Extracting from: {chunk.source_file} chunk {chunk.chunk_index}", end="\r")

        # Refresh known concept names periodically
        if chunks_processed > 0 and chunks_processed % refresh_known_every == 0:
            known_names = set(graph_client.get_all_concept_names())

        stats = ingest_text(chunk.text, graph_client, ollama_client, model)

        triples_total += stats["triples_extracted"]
        new_concepts_total += stats["new_concepts"]
        new_relationships_total += stats["new_relationships"]

        # Mark as processed and persist
        processed_ids.add(chunk.chunk_id)
        PROCESSED_CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PROCESSED_CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(list(processed_ids), f)

        chunks_processed += 1

    print()  # clear \r line
    return {
        "chunks_processed": chunks_processed,
        "chunks_skipped": chunks_skipped,
        "triples_total": triples_total,
        "new_concepts_total": new_concepts_total,
        "new_relationships_total": new_relationships_total,
    }
