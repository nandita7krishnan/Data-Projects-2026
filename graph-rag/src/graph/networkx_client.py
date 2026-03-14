"""
NetworkX-based graph client — default backend (no external DB needed).

Uses a directed multigraph (DiGraph) stored in-memory and persisted
to disk as a pickle file between sessions.
"""
from __future__ import annotations

import logging
import pickle
from typing import Optional

import networkx as nx

from config.settings import (
    GRAPH_HOPS,
    MAX_CONTEXT_NODES,
    NETWORKX_PERSIST_PATH,
    RELATIONSHIP_PRIORITY,
)
from src.graph.schema import ConceptNode, GraphSubgraph, Relationship

logger = logging.getLogger(__name__)


class NetworkXClient:
    """
    In-memory knowledge graph backed by NetworkX DiGraph.

    Node storage:  G.nodes[name] = ConceptNode.to_dict()
    Edge storage:  G.edges[src, dst, key]['rel_type'] = relationship type
    """

    def __init__(self, persist_path=NETWORKX_PERSIST_PATH):
        self.persist_path = persist_path
        self._graph: nx.MultiDiGraph = self._load()

    # ── Persistence ─────────────────────────────────────────────────────────

    def _load(self) -> nx.MultiDiGraph:
        if self.persist_path.exists():
            try:
                with open(self.persist_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning("Could not load graph from %s: %s", self.persist_path, e)
        return nx.MultiDiGraph()

    def _save(self) -> None:
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path, "wb") as f:
            pickle.dump(self._graph, f)

    # ── Write operations ────────────────────────────────────────────────────

    def upsert_concept(self, concept: ConceptNode) -> None:
        self._graph.add_node(concept.name, **concept.to_dict())
        self._save()

    def add_relationship(
        self,
        src: str,
        rel_type: str,
        dst: str,
        properties: dict = {},
    ) -> None:
        if src not in self._graph:
            logger.warning("Source node '%s' not in graph; skipping edge.", src)
            return
        if dst not in self._graph:
            logger.warning("Target node '%s' not in graph; skipping edge.", dst)
            return
        # Avoid duplicate edges of the same type
        for _, _, data in self._graph.edges(src, data=True):
            if data.get("rel_type") == rel_type and _ == dst:
                return
        self._graph.add_edge(src, dst, rel_type=rel_type, **properties)
        self._save()

    # ── Read operations ─────────────────────────────────────────────────────

    def _normalize(self, name: str) -> Optional[str]:
        """Return the canonical node name for a case-insensitive match."""
        name_lower = name.lower()
        for node in self._graph.nodes:
            if node.lower() == name_lower:
                return node
        return None

    def get_concept(self, name: str) -> Optional[ConceptNode]:
        canonical = self._normalize(name)
        if canonical is None:
            return None
        return ConceptNode.from_dict(dict(self._graph.nodes[canonical]))

    def get_neighbors(
        self,
        name: str,
        hops: int = GRAPH_HOPS,
        rel_types: Optional[list[str]] = None,
    ) -> GraphSubgraph:
        canonical = self._normalize(name)
        if canonical is None:
            raise ValueError(f"Concept '{name}' not found in graph.")

        # BFS — traverse both outgoing and incoming edges
        visited: dict[str, int] = {canonical: 0}
        queue: list[tuple[str, int]] = [(canonical, 0)]
        collected_edges: list[Relationship] = []

        while queue:
            current, depth = queue.pop(0)
            if depth >= hops:
                continue

            # Outgoing edges
            for _, dst, data in self._graph.edges(current, data=True):
                rt = data.get("rel_type", "RELATED")
                if rel_types and rt not in rel_types:
                    continue
                collected_edges.append(Relationship(current, rt, dst))
                if dst not in visited:
                    visited[dst] = depth + 1
                    queue.append((dst, depth + 1))

            # Incoming edges (predecessors)
            for pred in self._graph.predecessors(current):
                # MultiDiGraph: get all edges from pred → current
                edge_data = self._graph.get_edge_data(pred, current) or {}
                for _, data in edge_data.items():
                    rt = data.get("rel_type", "RELATED")
                    if rel_types and rt not in rel_types:
                        continue
                    collected_edges.append(Relationship(pred, rt, current))
                    if pred not in visited:
                        visited[pred] = depth + 1
                        queue.append((pred, depth + 1))

        # Sort by hop depth, then by relationship priority (most important first)
        sorted_pairs = sorted(visited.items(), key=lambda x: x[1])[:MAX_CONTEXT_NODES]
        hop_depth = dict(sorted_pairs)

        nodes = []
        for node_name, _ in sorted_pairs:
            c = self.get_concept(node_name)
            if c:
                nodes.append(c)

        # Deduplicate edges
        seen: set[tuple] = set()
        unique_edges: list[Relationship] = []
        for e in collected_edges:
            key = (e.source, e.rel_type, e.target)
            if key not in seen:
                seen.add(key)
                unique_edges.append(e)

        center = self.get_concept(canonical)
        return GraphSubgraph(
            center=center,
            nodes=nodes,
            relationships=unique_edges,
            hop_depth=hop_depth,
        )

    def search_concepts(self, query: str, limit: int = 10) -> list[ConceptNode]:
        """Score-ranked substring search on name and definition."""
        q = query.lower()
        results: list[tuple[int, str]] = []
        for node_name in self._graph.nodes:
            attrs = self._graph.nodes[node_name]
            score = 0
            if q == node_name.lower():
                score += 10
            elif q in node_name.lower():
                score += 5
            if q in attrs.get("definition", "").lower():
                score += 2
            if any(q in uc.lower() for uc in attrs.get("example_use_cases", [])):
                score += 1
            if score > 0:
                results.append((score, node_name))
        results.sort(reverse=True)
        return [
            self.get_concept(name)
            for _, name in results[:limit]
            if self.get_concept(name) is not None
        ]

    def get_all_concept_names(self) -> list[str]:
        return sorted(self._graph.nodes)

    def get_prerequisite_graph(self, name: str) -> GraphSubgraph:
        """Traversal restricted to prerequisite edges (REQUIRES, IS_A, PART_OF)."""
        return self.get_neighbors(
            name, hops=4, rel_types=["REQUIRES", "IS_A", "PART_OF"]
        )

    def get_stats(self) -> dict:
        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
            "backend": "networkx",
        }

    def get_nx_graph(self) -> nx.MultiDiGraph:
        """Expose the raw NetworkX graph for algorithms."""
        return self._graph

    def close(self) -> None:
        pass  # nothing to close for in-memory graph
