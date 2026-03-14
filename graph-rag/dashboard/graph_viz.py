"""
Interactive graph visualization using pyvis.

Renders the knowledge graph as an HTML network visualization
embeddable in Streamlit via st.components.v1.html().

Node properties:
  - Size: proportional to PageRank centrality score
  - Color: based on concept category
  - Label: concept name

Edge properties:
  - Label: relationship type
  - Arrow direction: source → target
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from config.settings import CATEGORY_COLORS
from src.graph.schema import ConceptNode, GraphSubgraph, Relationship


def build_full_graph_html(
    nodes: list[ConceptNode],
    relationships: list[Relationship],
    pagerank_scores: dict[str, float] = None,
    height: str = "600px",
    highlight: Optional[str] = None,
) -> str:
    """
    Build an interactive pyvis HTML graph of the full knowledge graph.

    Args:
        nodes: all concept nodes to display
        relationships: all edges to display
        pagerank_scores: {name: score} for sizing nodes
        height: CSS height string
        highlight: concept name to visually highlight

    Returns:
        HTML string to embed in Streamlit via st.components.v1.html()
    """
    try:
        from pyvis.network import Network
    except ImportError:
        return "<p>pyvis not installed. Run: pip install pyvis</p>"

    net = Network(
        height=height,
        width="100%",
        bgcolor="#0e1117",   # dark background matching Streamlit dark theme
        font_color="white",
        directed=True,
    )
    net.set_options(_get_physics_options())

    scores = pagerank_scores or {}
    max_score = max(scores.values()) if scores else 1.0

    # Add nodes
    added_nodes = set()
    for node in nodes:
        if node.name in added_nodes:
            continue
        added_nodes.add(node.name)

        color = CATEGORY_COLORS.get(node.category, "#888888")
        is_highlight = (node.name == highlight)

        # Scale node size between 15 and 50 based on PageRank
        raw_score = scores.get(node.name, 0.1)
        size = 15 + int((raw_score / max_score) * 35)

        if is_highlight:
            color = "#FFD700"  # gold for highlighted node
            size = max(size, 40)

        # Plain-text tooltip (pyvis renders this as-is in the browser)
        tooltip = (
            f"{node.name}\n"
            f"{node.category} | {node.difficulty}\n\n"
            f"{node.definition[:200]}..."
        )

        net.add_node(
            node.name,
            label=node.name,
            color=color,
            size=size,
            title=tooltip,
            font={"size": 12, "color": "white"},
            borderWidth=3 if is_highlight else 1,
            borderWidthSelected=4,
        )

    # Add edges
    for rel in relationships:
        if rel.source not in added_nodes or rel.target not in added_nodes:
            continue
        net.add_edge(
            rel.source,
            rel.target,
            label=rel.rel_type,
            title=f"{rel.source} {rel.rel_type} {rel.target}",
            color={"color": "#666666", "highlight": "#FFD700"},
            font={"size": 8, "color": "#aaaaaa"},
            arrows="to",
            smooth={"type": "curvedCW", "roundness": 0.2},
        )

    # Render to temp HTML and return as string
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        tmp_path = Path(f.name)
    net.save_graph(str(tmp_path))
    html = tmp_path.read_text(encoding="utf-8")
    tmp_path.unlink()
    return html


def build_subgraph_html(
    subgraph: GraphSubgraph,
    pagerank_scores: dict[str, float] = None,
    height: str = "450px",
) -> str:
    """Build a focused HTML graph for a single concept's neighborhood."""
    return build_full_graph_html(
        nodes=subgraph.nodes,
        relationships=subgraph.relationships,
        pagerank_scores=pagerank_scores,
        height=height,
        highlight=subgraph.center.name,
    )


def _get_physics_options() -> str:
    return """
    {
      "physics": {
        "enabled": true,
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "gravitationalConstant": -80,
          "centralGravity": 0.01,
          "springLength": 120,
          "springConstant": 0.08
        },
        "stabilization": {"iterations": 150}
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": true
      },
      "edges": {
        "smooth": {"type": "curvedCW", "roundness": 0.2},
        "font": {"size": 9}
      }
    }
    """
