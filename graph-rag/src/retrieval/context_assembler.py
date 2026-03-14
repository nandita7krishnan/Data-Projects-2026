"""
Context assembler — formats graph subgraph + optional vector results
into a structured markdown string for the LLM prompt.

This is where Graph RAG's core advantage shows:
the LLM receives structured relational context, not just text chunks.
"""
from __future__ import annotations

import logging
from typing import Optional

from config.settings import RELATIONSHIP_PRIORITY
from src.graph.schema import GraphSubgraph, Relationship

logger = logging.getLogger(__name__)

# Approximate token budget for context (leave room for LLM answer)
MAX_CONTEXT_CHARS = 6000


def assemble_context(
    subgraph: GraphSubgraph,
    vector_results: Optional[list[dict]] = None,
    include_definitions: bool = True,
) -> str:
    """
    Build a structured context string from a GraphSubgraph.

    Format:
        [CONCEPT: X]
        Definition: ...
        Category: ... | Difficulty: ...
        Example uses: ...

        [DIRECT RELATIONSHIPS]
        - REQUIRES → Vector Database: stores dense vectors...
        ...

        [RELATED CONCEPTS - HOP 2]
        - Transformer: a neural network architecture...
        ...

        [SUPPORTING TEXT - semantic search]
        ...
    """
    parts: list[str] = []

    # ── Center concept ───────────────────────────────────────────────────────
    center = subgraph.center
    parts.append(f"[CONCEPT: {center.name}]")
    parts.append(f"Definition: {center.definition}")
    parts.append(f"Category: {center.category} | Difficulty: {center.difficulty}")
    if center.example_use_cases:
        parts.append(f"Example uses: {', '.join(center.example_use_cases[:3])}")
    if center.related_tools:
        parts.append(f"Key tools: {', '.join(center.related_tools[:5])}")
    parts.append("")

    # ── Direct relationships (hop 1) ─────────────────────────────────────────
    direct_rels = _get_direct_relationships(subgraph)
    if direct_rels:
        parts.append("[DIRECT RELATIONSHIPS]")
        for rel in _sort_relationships(direct_rels):
            # Find the target/source node for its definition preview
            other_name = rel.target if rel.source == center.name else rel.source
            other_node = _find_node(subgraph, other_name)
            direction = "→" if rel.source == center.name else "←"
            preview = ""
            if other_node and include_definitions:
                preview = f": {other_node.short_description()}"
            parts.append(f"  - {rel.rel_type} {direction} {other_name}{preview}")
        parts.append("")

    # ── Related concepts by hop depth ────────────────────────────────────────
    max_hop = max(subgraph.hop_depth.values()) if subgraph.hop_depth else 0
    for hop in range(2, max_hop + 1):
        hop_nodes = subgraph.nodes_at_hop(hop)
        if not hop_nodes:
            continue
        parts.append(f"[RELATED CONCEPTS - HOP {hop}]")
        for node in hop_nodes[:8]:  # cap to avoid overly long context
            if include_definitions:
                parts.append(f"  - {node.name} ({node.category}): {node.short_description()}")
            else:
                parts.append(f"  - {node.name} ({node.category})")
        parts.append("")

    # ── Optional vector search results ───────────────────────────────────────
    if vector_results:
        parts.append("[SUPPORTING TEXT - semantic search]")
        for hit in vector_results[:3]:
            similarity = hit.get("similarity", 0)
            name = hit.get("metadata", {}).get("name", "")
            parts.append(f"  [{name} | similarity: {similarity:.2f}]")
            # Include first 300 chars of the document
            text_preview = hit["text"][:300].replace("\n", " ")
            parts.append(f"  {text_preview}")
            parts.append("")

    context = "\n".join(parts)

    # Truncate if too long
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n...[context truncated]"

    return context


def _get_direct_relationships(subgraph: GraphSubgraph) -> list[Relationship]:
    """Return relationships where center is source or target."""
    center_name = subgraph.center.name
    return [
        r for r in subgraph.relationships
        if r.source == center_name or r.target == center_name
    ]


def _sort_relationships(rels: list[Relationship]) -> list[Relationship]:
    """Sort by relationship priority (REQUIRES first, ALTERNATIVE_TO last)."""
    priority_map = {rt: i for i, rt in enumerate(RELATIONSHIP_PRIORITY)}
    return sorted(rels, key=lambda r: priority_map.get(r.rel_type, 99))


def _find_node(subgraph: GraphSubgraph, name: str):
    for node in subgraph.nodes:
        if node.name == name:
            return node
    return None
