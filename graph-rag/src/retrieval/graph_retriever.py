"""
Graph retriever — primary retrieval mechanism in the Graph RAG pipeline.

Given a concept name, retrieves its subgraph via BFS traversal and
returns a ranked, structured GraphSubgraph for context assembly.
"""
from __future__ import annotations

import logging
from typing import Optional

from config.settings import GRAPH_HOPS, MAX_CONTEXT_NODES
from src.graph.schema import GraphClient, GraphSubgraph

logger = logging.getLogger(__name__)


class GraphRetriever:
    """
    Wraps the GraphClient to provide retrieval modes:
      - standard: 2-hop BFS (default for general queries)
      - prerequisite: REQUIRES/IS_A only (for "what do I need to know?" queries)
      - similarity: SIMILAR_TO/VARIANT_OF/ALTERNATIVE_TO (for comparison queries)
    """

    def __init__(self, client: GraphClient, default_hops: int = GRAPH_HOPS):
        self._client = client
        self._default_hops = default_hops

    def retrieve(
        self,
        concept_name: str,
        mode: str = "standard",
        hops: Optional[int] = None,
    ) -> GraphSubgraph:
        """
        Main retrieval entry point.

        mode options:
          "standard"     — all relationship types, default hop depth
          "prerequisite" — REQUIRES/IS_A/PART_OF only (learning path)
          "similarity"   — SIMILAR_TO/VARIANT_OF/ALTERNATIVE_TO (comparisons)
          "usage"        — USED_IN/BUILT_WITH (applications)

        Returns GraphSubgraph with nodes ranked by hop depth.
        Raises ValueError if concept not found.
        """
        hop_depth = hops or self._default_hops
        rel_types = self._rel_types_for_mode(mode)

        logger.info(
            "Retrieving graph context for '%s' [mode=%s, hops=%d]",
            concept_name, mode, hop_depth,
        )

        subgraph = self._client.get_neighbors(
            concept_name,
            hops=hop_depth,
            rel_types=rel_types,
        )

        logger.info(
            "Retrieved %d nodes, %d relationships for '%s'",
            len(subgraph.nodes), len(subgraph.relationships), concept_name,
        )
        return subgraph

    def retrieve_prerequisites(self, concept_name: str) -> GraphSubgraph:
        """Get the full prerequisite chain (deep BFS on REQUIRES/IS_A/PART_OF)."""
        return self._client.get_prerequisite_graph(concept_name)

    def retrieve_direct_neighbors(self, concept_name: str) -> GraphSubgraph:
        """Get only direct (1-hop) neighbors — for quick concept cards."""
        return self._client.get_neighbors(concept_name, hops=1)

    # ── Query intent detection ───────────────────────────────────────────────

    @staticmethod
    def detect_mode(query: str) -> str:
        """
        Heuristic to detect the appropriate retrieval mode from query text.
        Returns one of: "standard", "prerequisite", "similarity", "usage"
        """
        q = query.lower()
        prerequisite_signals = [
            "before", "prerequisite", "require", "need to know", "depend",
            "foundation", "learn first", "should know", "start with",
        ]
        similarity_signals = [
            "similar", "compare", "difference", "versus", "vs", "alternative",
            "like", "same as", "related to",
        ]
        usage_signals = [
            "used in", "application", "example", "use case", "how is", "where is",
        ]

        if any(sig in q for sig in prerequisite_signals):
            return "prerequisite"
        if any(sig in q for sig in similarity_signals):
            return "similarity"
        if any(sig in q for sig in usage_signals):
            return "usage"
        return "standard"

    @staticmethod
    def _rel_types_for_mode(mode: str) -> Optional[list[str]]:
        modes = {
            "standard": None,  # all types
            "prerequisite": ["REQUIRES", "IS_A", "PART_OF"],
            "similarity": ["SIMILAR_TO", "VARIANT_OF", "ALTERNATIVE_TO"],
            "usage": ["USED_IN", "BUILT_WITH"],
        }
        return modes.get(mode, None)
