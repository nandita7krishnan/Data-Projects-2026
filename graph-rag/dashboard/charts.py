"""
Additional Plotly visualizations for the Graph RAG dashboard.

  - build_sunburst()  : radial hierarchy — category (inner) → concept (outer)
  - build_treemap()   : nested rectangles — category → concept, sized by PageRank
  - build_heatmap()   : category × category relationship count grid
"""
from __future__ import annotations

from collections import defaultdict

from config.settings import CATEGORY_COLORS
from src.graph.schema import ConceptNode, Relationship


# ── Sunburst ────────────────────────────────────────────────────────────────

def build_sunburst(
    concepts: list[ConceptNode],
    pagerank_scores: dict[str, float] = None,
):
    """
    Radial hierarchy chart.
    Inner ring = category, outer ring = individual concept.
    Sector size = PageRank score.  Parent value = sum of children (required for
    branchvalues='total' to render correctly).
    """
    import plotly.graph_objects as go

    scores = pagerank_scores or {}

    # Group concepts by category
    categories: dict[str, list[ConceptNode]] = {}
    for c in concepts:
        categories.setdefault(c.category, []).append(c)

    ids, labels, parents, values, colors, hovers = [], [], [], [], [], []

    # Compute bottom-up values so parents equal sum of children
    cat_totals: dict[str, float] = {}
    for cat, cat_concepts in categories.items():
        cat_totals[cat] = sum(scores.get(c.name, 0.01) for c in cat_concepts)

    root_total = sum(cat_totals.values())

    # Root
    ids.append("root")
    labels.append("AI Concepts")
    parents.append("")
    values.append(root_total)
    colors.append("#FFD6DC")
    hovers.append(f"<b>AI Concepts</b><br>{len(concepts)} concepts")

    for cat, cat_concepts in sorted(categories.items()):
        cat_color = CATEGORY_COLORS.get(cat, "#888888")

        ids.append(f"cat:{cat}")
        labels.append(cat)
        parents.append("root")
        values.append(cat_totals[cat])
        colors.append(cat_color)
        hovers.append(f"<b>{cat}</b><br>{len(cat_concepts)} concepts")

        for c in cat_concepts:
            score = scores.get(c.name, 0.01)
            defn = c.definition[:130] + "..." if len(c.definition) > 130 else c.definition
            ids.append(f"concept:{c.name}")
            labels.append(c.name)
            parents.append(f"cat:{cat}")
            values.append(score)
            colors.append(_lighten(cat_color, 0.35))
            hovers.append(
                f"<b>{c.name}</b><br>"
                f"<i>{c.difficulty}</i><br><br>"
                f"{defn}"
            )

    fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(colors=colors, line=dict(color="#FFE8EE", width=1.5)),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hovers,
        insidetextorientation="radial",
        maxdepth=3,
    ))

    fig.update_layout(
        paper_bgcolor="#FFE8EE",
        plot_bgcolor="#FFE8EE",
        margin=dict(t=40, b=10, l=10, r=10),
        height=560,
        font=dict(color="#3a1020", size=11),
        title=dict(
            text="AI Concepts — Radial Hierarchy by Category",
            font=dict(size=14, color="#3a1020"),
            x=0.5,
        ),
    )
    return fig


# ── Treemap ─────────────────────────────────────────────────────────────────

def build_treemap(
    concepts: list[ConceptNode],
    pagerank_scores: dict[str, float] = None,
):
    """
    Nested rectangle treemap.
    Parent = category, child = concept.
    Rectangle size = PageRank score.
    """
    import plotly.graph_objects as go

    scores = pagerank_scores or {}

    # Only include concepts that have a real PageRank score
    concepts = [c for c in concepts if scores.get(c.name, 0) > 0]

    categories: dict[str, list[ConceptNode]] = {}
    for c in concepts:
        categories.setdefault(c.category, []).append(c)

    cat_totals: dict[str, float] = {
        cat: sum(scores.get(c.name, 0.01) for c in cc)
        for cat, cc in categories.items()
    }
    root_total = sum(cat_totals.values())

    ids, labels, parents, values, colors, hovers = [], [], [], [], [], []

    # Root
    ids.append("root")
    labels.append("AI Concepts")
    parents.append("")
    values.append(root_total)
    colors.append("#FFD6DC")
    hovers.append(f"<b>AI Concepts</b><br>{len(concepts)} concepts")

    for cat, cat_concepts in sorted(categories.items()):
        cat_color = CATEGORY_COLORS.get(cat, "#888888")

        ids.append(f"cat:{cat}")
        labels.append(cat)
        parents.append("root")
        values.append(cat_totals[cat])
        colors.append(cat_color)
        hovers.append(f"<b>{cat}</b><br>{len(cat_concepts)} concepts")

        for c in cat_concepts:
            score = scores.get(c.name, 0.01)
            defn = c.definition[:100] + "..." if len(c.definition) > 100 else c.definition
            ids.append(f"concept:{c.name}")
            labels.append(c.name)
            parents.append(f"cat:{cat}")
            values.append(score)
            colors.append(_lighten(cat_color, 0.28))
            hovers.append(
                f"<b>{c.name}</b><br>"
                f"{c.category} | <i>{c.difficulty}</i><br><br>"
                f"{defn}"
            )

    fig = go.Figure(go.Treemap(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(
            colors=colors,
            line=dict(color="#FFE8EE", width=2),
        ),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hovers,
        textfont=dict(color="#3a1020", size=11),
        pathbar=dict(visible=True, thickness=22),
    ))

    fig.update_layout(
        paper_bgcolor="#FFE8EE",
        height=560,
        margin=dict(t=50, b=10, l=10, r=10),
        font=dict(color="#3a1020"),
        title=dict(
            text="AI Concepts — Treemap by Category  (size = PageRank importance)",
            font=dict(size=14, color="#3a1020"),
            x=0.5,
        ),
    )
    return fig


# ── Category Relationship Heatmap ────────────────────────────────────────────

def build_heatmap(
    concepts: list[ConceptNode],
    relationships: list[Relationship],
):
    """
    Category × Category relationship heatmap.
    Row = source category, Column = target category.
    Cell colour intensity = number of relationships between them.
    Hover shows the exact count and top relationship types.
    """
    import plotly.graph_objects as go

    concept_cat: dict[str, str] = {c.name: c.category for c in concepts}
    all_cats = sorted({c.category for c in concepts})

    # Count (src_cat, dst_cat) → {rel_type: count}
    flow: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for rel in relationships:
        src_cat = concept_cat.get(rel.source)
        dst_cat = concept_cat.get(rel.target)
        if src_cat and dst_cat:
            flow[(src_cat, dst_cat)][rel.rel_type] += 1

    n = len(all_cats)
    cat_idx = {cat: i for i, cat in enumerate(all_cats)}

    z = [[0.0] * n for _ in range(n)]
    hover = [[""] * n for _ in range(n)]

    for (src, dst), rel_counts in flow.items():
        i, j = cat_idx[src], cat_idx[dst]
        total = sum(rel_counts.values())
        z[i][j] = total
        top = sorted(rel_counts.items(), key=lambda x: -x[1])[:4]
        top_str = "<br>".join(f"  {r}: {cnt}" for r, cnt in top)
        hover[i][j] = (
            f"<b>{src} → {dst}</b><br>"
            f"Total: {total} relationships<br>"
            f"{top_str}"
        )

    # Fill empty cells with a placeholder hover
    for i, src in enumerate(all_cats):
        for j, dst in enumerate(all_cats):
            if not hover[i][j]:
                hover[i][j] = f"<b>{src} → {dst}</b><br>No direct relationships"

    fig = go.Figure(go.Heatmap(
        z=z,
        x=all_cats,
        y=all_cats,
        colorscale=[
            [0.0,  "#FFE8EE"],
            [0.01, "#FFB3BA"],
            [0.25, "#FF6B6B"],
            [0.6,  "#55A868"],
            [1.0,  "#C0392B"],
        ],
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover,
        showscale=True,
        colorbar=dict(
            title=dict(text="# Relationships", font=dict(color="#3a1020")),
            tickfont=dict(color="#3a1020"),
            bgcolor="#FFE8EE",
            bordercolor="#E8A0BF",
        ),
        xgap=2,
        ygap=2,
    ))

    fig.update_layout(
        paper_bgcolor="#FFE8EE",
        plot_bgcolor="#FFE8EE",
        height=520,
        margin=dict(t=60, b=80, l=120, r=20),
        font=dict(color="#3a1020", size=11),
        title=dict(
            text="Category × Category Relationship Heatmap",
            font=dict(size=14, color="#3a1020"),
            x=0.5,
        ),
        xaxis=dict(
            tickangle=-35,
            tickfont=dict(color="#3a1020", size=10),
            gridcolor="#FFD6DC",
        ),
        yaxis=dict(
            tickfont=dict(color="#3a1020", size=10),
            gridcolor="#FFD6DC",
            autorange="reversed",
        ),
    )
    return fig


# ── Helpers ──────────────────────────────────────────────────────────────────

def _lighten(hex_color: str, amount: float) -> str:
    """Blend a hex color toward white by `amount` (0–1)."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return "#" + hex_color
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"
