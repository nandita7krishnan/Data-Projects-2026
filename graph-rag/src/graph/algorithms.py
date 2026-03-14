"""
Advanced graph algorithms for knowledge retrieval.

Showcases graph theory applied to AI concept navigation:
  - PageRank: identify foundational/central concepts
  - BFS learning path: shortest path via prerequisite edges
  - Curriculum generation: topological sort of prerequisites
"""
from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Optional

import networkx as nx

if TYPE_CHECKING:
    from src.graph.networkx_client import NetworkXClient

logger = logging.getLogger(__name__)

# ── PageRank ────────────────────────────────────────────────────────────────

def get_concept_importance(graph_client: "NetworkXClient") -> dict[str, float]:
    """
    Run PageRank on the concept knowledge graph.

    Central concepts (Transformer, LLM, Embedding) naturally score higher
    because many other concepts point to or depend on them.

    Used to:
      - Rank search results when multiple matches exist
      - Size nodes in the graph visualization
      - Highlight 'foundational' concepts in the UI
    """
    G = graph_client.get_nx_graph()
    if G.number_of_nodes() == 0:
        return {}
    try:
        scores = nx.pagerank(G, alpha=0.85, max_iter=200)
        # Normalize to [0, 1]
        max_score = max(scores.values()) if scores else 1.0
        return {k: v / max_score for k, v in scores.items()}
    except Exception as e:
        logger.warning("PageRank failed: %s", e)
        return {n: 1.0 for n in G.nodes}


# ── BFS Learning Path ───────────────────────────────────────────────────────

PREREQUISITE_RELS = {"REQUIRES", "IS_A", "PART_OF"}

def find_learning_path(
    graph_client: "NetworkXClient",
    from_concept: str,
    to_concept: str,
) -> list[str]:
    """
    Find the shortest learning path from `from_concept` to `to_concept`.

    Strategy:
      1. BFS restricted to REQUIRES / IS_A / PART_OF edges (prerequisite graph)
      2. If no path found, fall back to BFS on all edges
      3. If still no path, return [from_concept, to_concept] as a hint

    Returns an ordered list of concept names representing the path.

    Example: find_learning_path(G, "Python", "GraphRAG")
      → ["Python", "LLM", "RAG", "Knowledge Graph", "GraphRAG"]
    """
    G = graph_client.get_nx_graph()

    # Try prerequisite-only path first
    path = _bfs_path(G, from_concept, to_concept, rel_filter=PREREQUISITE_RELS)
    if path:
        return path

    # Fall back to all edges
    path = _bfs_path(G, from_concept, to_concept, rel_filter=None)
    if path:
        return path

    logger.info(
        "No path from '%s' to '%s'; returning direct suggestion.",
        from_concept, to_concept,
    )
    return [from_concept, to_concept]


def _bfs_path(
    G: nx.MultiDiGraph,
    src: str,
    dst: str,
    rel_filter: Optional[set[str]],
) -> list[str]:
    """BFS on `G` from `src` to `dst`, optionally filtering edge types."""
    # Normalise names (case-insensitive)
    node_map = {n.lower(): n for n in G.nodes}
    src_canon = node_map.get(src.lower())
    dst_canon = node_map.get(dst.lower())
    if src_canon is None or dst_canon is None:
        return []

    visited = {src_canon: None}  # node → parent
    queue = deque([src_canon])

    while queue:
        current = queue.popleft()
        if current == dst_canon:
            return _reconstruct_path(visited, dst_canon)

        # Traverse outgoing + incoming edges for flexibility
        neighbors: list[str] = []
        for _, tgt, data in G.edges(current, data=True):
            rt = data.get("rel_type", "")
            if rel_filter is None or rt in rel_filter:
                neighbors.append(tgt)
        for pred in G.predecessors(current):
            edge_data = G.get_edge_data(pred, current) or {}
            for _, data in edge_data.items():
                rt = data.get("rel_type", "")
                if rel_filter is None or rt in rel_filter:
                    neighbors.append(pred)
                    break  # one edge type is enough to include the neighbor

        for n in neighbors:
            if n not in visited:
                visited[n] = current
                queue.append(n)

    return []


def _reconstruct_path(parent: dict[str, Optional[str]], dst: str) -> list[str]:
    path = []
    current: Optional[str] = dst
    while current is not None:
        path.append(current)
        current = parent[current]
    return list(reversed(path))


# ── Curriculum Generator ────────────────────────────────────────────────────

def generate_curriculum(
    graph_client: "NetworkXClient",
    target: str,
    known: list[str],
) -> list[str]:
    """
    Generate an ordered learning curriculum to reach `target` concept.

    Algorithm:
      1. Walk the prerequisite subgraph (REQUIRES, IS_A, PART_OF) from `target`
      2. Collect all prerequisite concepts
      3. Topological sort the collected subgraph
      4. Filter out already-known concepts
      5. Return in learning order (prerequisites first)

    Example:
      generate_curriculum(G, "GraphRAG", known=["Python", "Machine Learning"])
      → ["Transformer", "LLM", "Embedding", "Vector Database",
         "Knowledge Graph", "RAG", "GraphRAG"]
    """
    G = graph_client.get_nx_graph()
    known_lower = {k.lower() for k in known}

    # BFS to collect all prerequisites of target
    target_canon = _normalize_name(G, target)
    if target_canon is None:
        return [target]

    prereqs = _collect_prerequisites(G, target_canon)

    # Build a subgraph with only prerequisite relationships
    sub_nodes = prereqs | {target_canon}
    sub = nx.DiGraph()
    sub.add_nodes_from(sub_nodes)

    for node in sub_nodes:
        for _, tgt, data in G.edges(node, data=True):
            rt = data.get("rel_type", "")
            if rt in PREREQUISITE_RELS and tgt in sub_nodes:
                # edge: tgt → node means "node depends on tgt" (tgt is prerequisite of node)
                sub.add_edge(tgt, node)

    # Topological sort — concepts with no prerequisites come first
    try:
        ordered = list(nx.topological_sort(sub))
    except nx.NetworkXUnfeasible:
        ordered = list(sub_nodes)

    # Filter known concepts, keep target last
    curriculum = [
        c for c in ordered
        if c.lower() not in known_lower
    ]

    # Ensure target is at the end
    if target_canon in curriculum:
        curriculum.remove(target_canon)
    curriculum.append(target_canon)

    return curriculum


def _collect_prerequisites(G: nx.MultiDiGraph, start: str) -> set[str]:
    """
    BFS forward through prerequisite edges from `start`.

    Follows outgoing REQUIRES/IS_A/PART_OF edges: if A REQUIRES B,
    then B is a prerequisite of A and should appear earlier in the curriculum.
    """
    visited = set()
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for _, successor, data in G.edges(current, data=True):
            rt = data.get("rel_type", "")
            if rt in PREREQUISITE_RELS and successor not in visited:
                visited.add(successor)
                queue.append(successor)
    return visited


def _normalize_name(G: nx.MultiDiGraph, name: str) -> Optional[str]:
    name_lower = name.lower()
    for node in G.nodes:
        if node.lower() == name_lower:
            return node
    return None
