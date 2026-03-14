"""
Graph client factory.

Usage:
    from src.graph import get_graph_client
    client = get_graph_client()

The client returned implements the GraphClient protocol.
Backend is chosen via GRAPH_BACKEND env var (default: "networkx").
Falls back to NetworkX automatically if Neo4j is unavailable.
"""
import logging
from typing import Optional

from config.settings import GRAPH_BACKEND
from src.graph.schema import GraphClient

logger = logging.getLogger(__name__)

_client_instance = None  # type: Optional[GraphClient]


def get_graph_client(force_new: bool = False) -> GraphClient:
    """
    Return the configured graph backend (singleton by default).
    Pass force_new=True to create a fresh client (useful in tests).
    """
    global _client_instance
    if _client_instance is not None and not force_new:
        return _client_instance

    if GRAPH_BACKEND == "neo4j":
        try:
            from src.graph.neo4j_client import Neo4jClient
            _client_instance = Neo4jClient()
            logger.info("Using Neo4j graph backend.")
            return _client_instance
        except Exception as e:
            logger.warning(
                "Neo4j unavailable (%s). Falling back to NetworkX.", e
            )

    from src.graph.networkx_client import NetworkXClient
    _client_instance = NetworkXClient()
    logger.info("Using NetworkX graph backend.")
    return _client_instance


def reset_client() -> None:
    """Reset the singleton (useful in tests)."""
    global _client_instance
    if _client_instance is not None:
        _client_instance.close()
    _client_instance = None
