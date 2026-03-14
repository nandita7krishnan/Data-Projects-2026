"""
Core data model for the Graph RAG knowledge graph.

Defines:
  - ConceptNode: a node in the AI concept graph
  - Relationship: a directed edge between two concepts
  - GraphSubgraph: result of a graph traversal query
  - GraphClient: Protocol that both NetworkX and Neo4j backends implement
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable


# ── Relationship types ──────────────────────────────────────────────────────

RELATIONSHIP_TYPES = [
    "IS_A",           # RAG IS_A Language Model Application
    "PART_OF",        # Attention Mechanism PART_OF Transformer
    "VARIANT_OF",     # GraphRAG VARIANT_OF RAG
    "SIMILAR_TO",     # FAISS SIMILAR_TO ChromaDB  (symmetric)
    "USED_IN",        # Embedding USED_IN Semantic Search
    "REQUIRES",       # RAG REQUIRES Vector Database
    "ALTERNATIVE_TO", # LoRA ALTERNATIVE_TO Fine-tuning  (symmetric)
    "BUILT_WITH",     # LangChain BUILT_WITH LLM
    "EXTENDS",        # ReAct EXTENDS Chain-of-Thought
]

DIFFICULTY_LEVELS = ["beginner", "intermediate", "advanced", "expert"]

CATEGORIES = [
    "Architecture", "Technique", "Model", "Tool", "Library",
    "Framework", "Concept", "Algorithm", "Dataset", "Evaluation",
    "Domain", "Infrastructure",
]


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class ConceptNode:
    """A node in the AI concept knowledge graph."""
    name: str
    definition: str                          # beginner-friendly, 2-3 sentences
    category: str                            # one of CATEGORIES
    difficulty: str                          # one of DIFFICULTY_LEVELS
    example_use_cases: list[str] = field(default_factory=list)
    related_tools: list[str] = field(default_factory=list)
    source_references: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "definition": self.definition,
            "category": self.category,
            "difficulty": self.difficulty,
            "example_use_cases": self.example_use_cases,
            "related_tools": self.related_tools,
            "source_references": self.source_references,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConceptNode":
        return cls(
            name=d["name"],
            definition=d["definition"],
            category=d.get("category", "Concept"),
            difficulty=d.get("difficulty", "intermediate"),
            example_use_cases=d.get("example_use_cases", []),
            related_tools=d.get("related_tools", []),
            source_references=d.get("source_references", []),
        )

    def short_description(self) -> str:
        """First sentence of the definition."""
        sentences = self.definition.split(". ")
        return sentences[0] + "." if sentences else self.definition


@dataclass
class Relationship:
    """A directed edge in the knowledge graph."""
    source: str       # concept name
    rel_type: str     # one of RELATIONSHIP_TYPES
    target: str       # concept name
    properties: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"({self.source}) -[{self.rel_type}]-> ({self.target})"


@dataclass
class GraphSubgraph:
    """Result of a graph traversal query — a subgraph centered on one concept."""
    center: ConceptNode
    nodes: list[ConceptNode]            # all retrieved nodes (includes center)
    relationships: list[Relationship]   # all edges within this subgraph
    hop_depth: dict[str, int]           # name → hop distance from center

    def nodes_at_hop(self, hop: int) -> list[ConceptNode]:
        names = {n for n, h in self.hop_depth.items() if h == hop}
        return [node for node in self.nodes if node.name in names]

    def relationships_for(self, concept_name: str) -> list[Relationship]:
        return [
            r for r in self.relationships
            if r.source == concept_name or r.target == concept_name
        ]


# ── GraphClient Protocol ────────────────────────────────────────────────────

@runtime_checkable
class GraphClient(Protocol):
    """
    Abstract interface for the graph backend.
    Implemented by NetworkXClient and Neo4jClient.
    """

    def upsert_concept(self, concept: ConceptNode) -> None:
        """Insert or update a concept node."""
        ...

    def add_relationship(
        self,
        src: str,
        rel_type: str,
        dst: str,
        properties: dict = {},
    ) -> None:
        """Add a directed relationship between two concept names."""
        ...

    def get_concept(self, name: str) -> Optional[ConceptNode]:
        """Retrieve a concept by name (case-insensitive). Returns None if not found."""
        ...

    def get_neighbors(
        self,
        name: str,
        hops: int = 2,
        rel_types: Optional[list[str]] = None,
    ) -> GraphSubgraph:
        """
        BFS traversal up to `hops` from `name`.
        Optionally filter by relationship types.
        """
        ...

    def search_concepts(self, query: str, limit: int = 10) -> list[ConceptNode]:
        """Fuzzy search by name and definition substring."""
        ...

    def get_all_concept_names(self) -> list[str]:
        """Return all concept names in the graph."""
        ...

    def get_prerequisite_graph(self, name: str) -> GraphSubgraph:
        """Traversal restricted to REQUIRES, IS_A, PART_OF edges (prerequisite chain)."""
        ...

    def get_stats(self) -> dict:
        """Return graph statistics: node count, edge count, etc."""
        ...

    def close(self) -> None:
        """Release any resources (connections, file handles)."""
        ...
