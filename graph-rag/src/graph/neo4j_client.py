"""
Neo4j-based graph client — optional backend.

Requires a running Neo4j instance. Configure via .env:
  GRAPH_BACKEND=neo4j
  NEO4J_URI=bolt://localhost:7687
  NEO4J_USER=neo4j
  NEO4J_PASSWORD=password
"""
from __future__ import annotations

import logging
from typing import Optional

from neo4j import GraphDatabase, Driver

from config.settings import (
    GRAPH_HOPS,
    MAX_CONTEXT_NODES,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
)
from src.graph.schema import ConceptNode, GraphSubgraph, Relationship

logger = logging.getLogger(__name__)


class Neo4jClient:
    """
    Knowledge graph backed by Neo4j.
    All concept nodes carry the label :Concept.
    Relationship types are dynamic (one of RELATIONSHIP_TYPES).
    """

    def __init__(self):
        self._driver: Driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        self._ensure_constraints()

    def _ensure_constraints(self) -> None:
        with self._driver.session() as session:
            session.run(
                "CREATE CONSTRAINT concept_name IF NOT EXISTS "
                "FOR (c:Concept) REQUIRE c.name IS UNIQUE"
            )

    # ── Write operations ────────────────────────────────────────────────────

    def upsert_concept(self, concept: ConceptNode) -> None:
        with self._driver.session() as session:
            session.run(
                """
                MERGE (c:Concept {name: $name})
                SET c.definition       = $definition,
                    c.category         = $category,
                    c.difficulty       = $difficulty,
                    c.example_use_cases = $example_use_cases,
                    c.related_tools    = $related_tools,
                    c.source_references = $source_references
                """,
                **concept.to_dict(),
            )

    def add_relationship(
        self,
        src: str,
        rel_type: str,
        dst: str,
        properties: dict = {},
    ) -> None:
        # Dynamically create typed relationship
        query = f"""
        MATCH (a:Concept {{name: $src}}), (b:Concept {{name: $dst}})
        MERGE (a)-[r:{rel_type}]->(b)
        """
        with self._driver.session() as session:
            session.run(query, src=src, dst=dst)

    # ── Read operations ─────────────────────────────────────────────────────

    def get_concept(self, name: str) -> Optional[ConceptNode]:
        with self._driver.session() as session:
            result = session.run(
                "MATCH (c:Concept) WHERE toLower(c.name) = toLower($name) RETURN c",
                name=name,
            )
            record = result.single()
            if record is None:
                return None
            return ConceptNode.from_dict(dict(record["c"]))

    def get_neighbors(
        self,
        name: str,
        hops: int = GRAPH_HOPS,
        rel_types: Optional[list[str]] = None,
    ) -> GraphSubgraph:
        center = self.get_concept(name)
        if center is None:
            raise ValueError(f"Concept '{name}' not found in graph.")

        rel_filter = ""
        if rel_types:
            joined = "|".join(rel_types)
            rel_filter = f":{joined}"

        query = f"""
        MATCH path = (c:Concept {{name: $name}})-[{rel_filter}*1..{hops}]-(n:Concept)
        RETURN n, relationships(path) AS rels, length(path) AS depth
        LIMIT $limit
        """
        nodes: list[ConceptNode] = [center]
        hop_depth: dict[str, int] = {name: 0}
        rels: list[Relationship] = []

        with self._driver.session() as session:
            results = session.run(query, name=name, limit=MAX_CONTEXT_NODES)
            for record in results:
                node = ConceptNode.from_dict(dict(record["n"]))
                if node.name not in hop_depth:
                    hop_depth[node.name] = record["depth"]
                    nodes.append(node)
                for r in record["rels"]:
                    rel = Relationship(
                        source=r.start_node["name"],
                        rel_type=r.type,
                        target=r.end_node["name"],
                    )
                    if rel not in rels:
                        rels.append(rel)

        return GraphSubgraph(
            center=center,
            nodes=nodes,
            relationships=rels,
            hop_depth=hop_depth,
        )

    def search_concepts(self, query: str, limit: int = 10) -> list[ConceptNode]:
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (c:Concept)
                WHERE toLower(c.name) CONTAINS toLower($query)
                   OR toLower(c.definition) CONTAINS toLower($query)
                RETURN c LIMIT $limit
                """,
                query=query,
                limit=limit,
            )
            return [ConceptNode.from_dict(dict(r["c"])) for r in result]

    def get_all_concept_names(self) -> list[str]:
        with self._driver.session() as session:
            result = session.run("MATCH (c:Concept) RETURN c.name AS name ORDER BY name")
            return [r["name"] for r in result]

    def get_prerequisite_graph(self, name: str) -> GraphSubgraph:
        return self.get_neighbors(
            name, hops=4, rel_types=["REQUIRES", "IS_A", "PART_OF"]
        )

    def get_stats(self) -> dict:
        with self._driver.session() as session:
            nodes = session.run("MATCH (c:Concept) RETURN count(c) AS n").single()["n"]
            edges = session.run("MATCH ()-[r]->() RETURN count(r) AS n").single()["n"]
        return {"nodes": nodes, "edges": edges, "backend": "neo4j"}

    def close(self) -> None:
        self._driver.close()
