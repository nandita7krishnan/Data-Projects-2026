"""
Community detector — runs Louvain community detection on the knowledge graph
to identify clusters of closely related concepts.

These communities are the foundation of GraphRAG's global retrieval:
each community gets an LLM-generated summary, and global queries
map-reduce over those summaries.

Usage:
    from src.graph.community_detector import detect_communities
    communities = detect_communities(graph_client)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from config.settings import (
    COMMUNITIES_PATH,
    COMMUNITY_RESOLUTION,
    MIN_COMMUNITY_SIZE,
)
from src.graph.schema import Community

logger = logging.getLogger(__name__)


def detect_communities(
    graph_client,
    resolution: float = COMMUNITY_RESOLUTION,
    min_size: int = MIN_COMMUNITY_SIZE,
    persist: bool = True,
) -> list[Community]:
    """
    Run Louvain community detection on the knowledge graph.

    Converts the directed multigraph to undirected (required by Louvain),
    detects communities, filters out small clusters, and returns them
    sorted by size descending.

    Args:
        graph_client: NetworkXClient (must have get_nx_graph())
        resolution: Louvain resolution parameter (higher = more, smaller communities)
        min_size: minimum community size to keep
        persist: save results to COMMUNITIES_PATH

    Returns:
        list of Community objects, sorted by size descending
    """
    try:
        import community as community_louvain
    except ImportError:
        raise ImportError(
            "python-louvain is required for community detection.\n"
            "Install it with: pip install python-louvain"
        )

    # Get the raw NetworkX graph
    G = graph_client.get_nx_graph()

    if G.number_of_nodes() == 0:
        logger.warning("Graph is empty. Run `python main.py seed` or `python main.py ingest` first.")
        return []

    # Louvain works on undirected graphs
    G_undirected = G.to_undirected()

    logger.info(
        "Running Louvain on %d nodes, %d edges (resolution=%.2f)...",
        G_undirected.number_of_nodes(),
        G_undirected.number_of_edges(),
        resolution,
    )

    # partition: {node_name: community_id}
    partition = community_louvain.best_partition(G_undirected, resolution=resolution)

    # Group nodes by community id
    groups: dict[int, list[str]] = {}
    for node_name, comm_id in partition.items():
        groups.setdefault(comm_id, []).append(node_name)

    # Build Community objects, filter small ones, re-index sequentially
    communities: list[Community] = []
    new_id = 0
    for comm_id, node_names in sorted(groups.items(), key=lambda x: len(x[1]), reverse=True):
        if len(node_names) < min_size:
            continue
        communities.append(
            Community(
                community_id=new_id,
                node_names=sorted(node_names),
            )
        )
        new_id += 1

    logger.info(
        "Detected %d communities (≥%d nodes) from %d raw partitions.",
        len(communities), min_size, len(groups),
    )

    if persist:
        _save_communities(communities)

    return communities


def load_communities(path: Path = COMMUNITIES_PATH) -> list[Community]:
    """Load communities from the persisted JSON file."""
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return [Community.from_dict(d) for d in data]
    except Exception as e:
        logger.warning("Could not load communities from %s: %s", path, e)
        return []


def _save_communities(communities: list[Community], path: Path = COMMUNITIES_PATH) -> None:
    """Persist communities to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in communities], f, indent=2)
    logger.info("Saved %d communities to %s", len(communities), path)
