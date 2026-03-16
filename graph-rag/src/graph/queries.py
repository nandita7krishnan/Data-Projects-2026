"""
Named traversal helpers — convenience wrappers over GraphClient.

These represent common query patterns in the Graph RAG pipeline:
  - What does concept X depend on?
  - What concepts are similar to X?
  - What is X a part of?
  - How do X and Y compare?
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from src.graph.schema import ConceptNode, GraphSubgraph

if TYPE_CHECKING:
    from src.graph.schema import GraphClient


def get_dependencies(client: "GraphClient", concept_name: str) -> GraphSubgraph:
    """
    Return all concepts that `concept_name` directly or transitively depends on.
    Traverses REQUIRES, IS_A, PART_OF edges up to 4 hops.
    """
    return client.get_neighbors(
        concept_name,
        hops=4,
        rel_types=["REQUIRES", "IS_A", "PART_OF"],
    )


def get_similar_concepts(client: "GraphClient", concept_name: str) -> list[ConceptNode]:
    """
    Return concepts that are SIMILAR_TO, ALTERNATIVE_TO, or COMPLEMENTARY_TO `concept_name`.
    """
    subgraph = client.get_neighbors(
        concept_name,
        hops=1,
        rel_types=["SIMILAR_TO", "ALTERNATIVE_TO", "COMPLEMENTARY_TO", "VARIANT_OF"],
    )
    return [n for n in subgraph.nodes if n.name != concept_name]


def get_use_cases(client: "GraphClient", concept_name: str) -> list[ConceptNode]:
    """
    Return concepts that USE or are BUILT_WITH `concept_name`.
    """
    subgraph = client.get_neighbors(
        concept_name,
        hops=2,
        rel_types=["USED_IN", "BUILT_WITH"],
    )
    return [n for n in subgraph.nodes if n.name != concept_name]


def get_comparison(
    client: "GraphClient",
    concept_a: str,
    concept_b: str,
) -> dict:
    """
    Build a comparison dict between two concepts.
    Returns their node data plus shared prerequisites.
    """
    node_a = client.get_concept(concept_a)
    node_b = client.get_concept(concept_b)

    if node_a is None or node_b is None:
        missing = concept_a if node_a is None else concept_b
        raise ValueError(f"Concept '{missing}' not found in graph.")

    # Find shared prerequisites by intersecting 2-hop neighbor sets
    neighbors_a = {
        n.name
        for n in client.get_neighbors(concept_a, hops=2, rel_types=["REQUIRES", "IS_A"]).nodes
        if n.name != concept_a
    }
    neighbors_b = {
        n.name
        for n in client.get_neighbors(concept_b, hops=2, rel_types=["REQUIRES", "IS_A"]).nodes
        if n.name != concept_b
    }

    shared = neighbors_a & neighbors_b
    only_a = neighbors_a - neighbors_b
    only_b = neighbors_b - neighbors_a

    return {
        "concept_a": node_a.to_dict(),
        "concept_b": node_b.to_dict(),
        "shared_prerequisites": sorted(shared),
        "unique_to_a": sorted(only_a),
        "unique_to_b": sorted(only_b),
    }
