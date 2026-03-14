"""
Seed the knowledge graph and vector store from data/seed_concepts.json.

Run via:
    python main.py seed
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from tqdm import tqdm

from config.settings import SEED_DATA_PATH
from src.graph.schema import ConceptNode, GraphClient

logger = logging.getLogger(__name__)


def load_seed_concepts(path: Path = SEED_DATA_PATH) -> tuple[list[ConceptNode], list[dict]]:
    """Parse seed JSON → (list of ConceptNode, list of relationship dicts)."""
    with open(path) as f:
        data = json.load(f)

    concepts = [ConceptNode.from_dict(c) for c in data["concepts"]]
    relationships = data.get("relationships", [])
    return concepts, relationships


def seed_graph(client: GraphClient, path: Path = SEED_DATA_PATH) -> dict:
    """
    Load all concepts and relationships from seed JSON into the graph client.
    Returns stats dict: {concepts_added, relationships_added, skipped}.
    """
    concepts, relationships = load_seed_concepts(path)

    logger.info("Seeding graph with %d concepts...", len(concepts))
    for concept in tqdm(concepts, desc="Adding concept nodes"):
        client.upsert_concept(concept)

    skipped = 0
    logger.info("Adding %d relationships...", len(relationships))
    all_names = set(client.get_all_concept_names())

    for rel in tqdm(relationships, desc="Adding relationships"):
        src = rel["source"]
        dst = rel["target"]
        rt = rel["rel_type"]

        if src not in all_names:
            logger.debug("Skipping rel: source '%s' not in graph.", src)
            skipped += 1
            continue
        if dst not in all_names:
            logger.debug("Skipping rel: target '%s' not in graph.", dst)
            skipped += 1
            continue

        client.add_relationship(src, rt, dst)

    stats = {
        "concepts_added": len(concepts),
        "relationships_added": len(relationships) - skipped,
        "skipped_relationships": skipped,
    }
    logger.info("Graph seeding complete: %s", stats)
    return stats


def seed_vector_store(
    concepts: list[ConceptNode],
    vector_retriever=None,
) -> int:
    """
    Index all concepts into ChromaDB for supplementary semantic search.
    Returns the number of documents indexed.

    `vector_retriever` should be an instance of VectorRetriever.
    If None, this step is skipped silently.
    """
    if vector_retriever is None:
        logger.info("No vector retriever provided; skipping vector indexing.")
        return 0

    documents = []
    ids = []
    metadatas = []

    for concept in concepts:
        # Build a rich document string for better semantic matching
        doc = (
            f"{concept.name}\n"
            f"{concept.definition}\n"
            f"Category: {concept.category}\n"
            f"Difficulty: {concept.difficulty}\n"
            f"Use cases: {', '.join(concept.example_use_cases)}"
        )
        documents.append(doc)
        ids.append(concept.name.lower().replace(" ", "_"))
        metadatas.append({
            "name": concept.name,
            "category": concept.category,
            "difficulty": concept.difficulty,
        })

    vector_retriever.add_documents(documents, ids=ids, metadatas=metadatas)
    logger.info("Indexed %d concepts into vector store.", len(documents))
    return len(documents)
