"""
Streamlit dashboard for the Graph RAG AI Concepts Explorer.

Four tabs:
  1. Concept Search  — look up any AI term, get a clear definition + graph answer
  2. Graph Explorer  — interactive pyvis knowledge graph visualization
  3. Learning Path   — generate a personalized learning curriculum
  4. Compare         — side-by-side concept comparison

Run with:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from config.settings import CATEGORY_COLORS, DIFFICULTY_COLORS, OLLAMA_MODEL
from src.graph import get_graph_client
from src.graph.algorithms import (
    find_learning_path,
    generate_curriculum,
    get_concept_importance,
)
from src.graph.schema import ConceptNode
from src.llm.answer_generator import AnswerGenerator
from src.retrieval.context_assembler import assemble_context
from src.retrieval.entity_extractor import EntityExtractor
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.vector_retriever import VectorRetriever

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Concepts Explorer | Graph RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── App-wide pastel pink background + dark text ── */
.stApp, [data-testid="stAppViewContainer"] {
    background-color: #FFE8EE !important;
    color: #3a1020 !important;
}
[data-testid="stSidebar"] {
    background-color: #FFD6DC !important;
    color: #3a1020 !important;
}
[data-testid="stSidebar"] * {
    color: #3a1020 !important;
}
[data-testid="stHeader"] {
    background-color: #FFE8EE !important;
}
/* Force all text dark */
h1, h2, h3, h4, h5, h6,
p, span, div, label, li,
.stMarkdown, .stMarkdown p,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
.stText, .stCaption,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] * {
    color: #3a1020 !important;
}
/* Tab labels */
button[data-baseweb="tab"] {
    color: #3a1020 !important;
}
/* Input / select boxes */
.stTextInput input, .stSelectbox select,
[data-baseweb="input"] input,
[data-baseweb="select"] * {
    color: #3a1020 !important;
    background-color: #FFF0F3 !important;
}
/* Metric labels */
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
    color: #3a1020 !important;
}
/* Expander */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span {
    color: #3a1020 !important;
}
.concept-card {
    background: #FFF0F3;
    border-radius: 10px;
    padding: 16px;
    margin: 8px 0;
    border-left: 4px solid #E8A0BF;
    color: #3a1020;
}
.difficulty-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: bold;
    color: white;
}
.rel-tag {
    display: inline-block;
    background: #FFD6DC;
    border-radius: 6px;
    padding: 2px 8px;
    margin: 2px;
    font-size: 12px;
    color: #8B3045;
}
</style>
""", unsafe_allow_html=True)


# ── Cached resource loading ──────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading knowledge graph...")
def load_resources():
    client = get_graph_client()
    vector = VectorRetriever()
    generator = AnswerGenerator()
    names = client.get_all_concept_names()
    extractor = EntityExtractor(
        names,
        ollama_client=generator.get_client(),
        model=OLLAMA_MODEL,
    )
    retriever = GraphRetriever(client)
    return client, vector, generator, extractor, retriever


@st.cache_data(ttl=300, show_spinner=False)
def get_pagerank():
    client = get_graph_client()
    try:
        from src.graph.networkx_client import NetworkXClient
        if isinstance(client, NetworkXClient):
            return get_concept_importance(client)
    except Exception:
        pass
    return {}


@st.cache_data(ttl=60, show_spinner=False)
def get_all_concepts():
    client = get_graph_client()
    names = client.get_all_concept_names()
    concepts = [client.get_concept(n) for n in names]
    return [c for c in concepts if c is not None]


client, vector_retriever, answer_generator, entity_extractor, graph_retriever = load_resources()
pagerank_scores = get_pagerank()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 AI Concepts Explorer")
    st.caption("Powered by Graph RAG")

    stats = client.get_stats()
    col1, col2 = st.columns(2)
    col1.metric("Concepts", stats["nodes"])
    col2.metric("Relationships", stats["edges"])

    st.divider()
    st.caption("**Graph Backend:** " + stats["backend"].upper())
    if answer_generator.available:
        st.success(f"✓ Ollama ({OLLAMA_MODEL})")
    else:
        st.warning("⚠ Ollama offline — graph context only")

    if vector_retriever.available:
        st.success(f"✓ Vector store ({vector_retriever.count()} docs)")
    else:
        st.info("ℹ Vector search disabled")

    st.divider()
    st.caption("**Legend — Categories**")
    for cat, color in list(CATEGORY_COLORS.items())[:6]:
        st.markdown(
            f'<span style="color:{color}">■</span> {cat}',
            unsafe_allow_html=True,
        )


# ── Tab layout ───────────────────────────────────────────────────────────────
tab_search, tab_graph, tab_path, tab_compare = st.tabs([
    "🔍 Concept Search",
    "🕸 Graph Explorer",
    "🎓 Learning Path",
    "⚖️ Compare Concepts",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: CONCEPT SEARCH
# ═══════════════════════════════════════════════════════════════════════════
with tab_search:
    st.header("Concept Search")
    st.caption(
        "Type any AI term to get a clear definition, see its relationships, "
        "and ask questions about it."
    )

    col_search, col_mode = st.columns([3, 1])
    with col_search:
        all_names = client.get_all_concept_names()
        query = st.text_input(
            "Search or ask a question",
            placeholder='e.g. "What is RAG?" or "What does GraphRAG depend on?"',
            label_visibility="collapsed",
        )
    with col_mode:
        mode_options = ["auto-detect", "standard", "prerequisite", "similarity", "usage"]
        selected_mode = st.selectbox("Mode", mode_options, label_visibility="collapsed")

    # Quick concept selector
    with st.expander("Browse all concepts", expanded=False):
        concepts_all = get_all_concepts()
        cat_filter = st.multiselect(
            "Filter by category",
            options=sorted(CATEGORY_COLORS.keys()),
        )
        filtered = [c for c in concepts_all if not cat_filter or c.category in cat_filter]
        filtered_sorted = sorted(filtered, key=lambda c: pagerank_scores.get(c.name, 0), reverse=True)

        cols = st.columns(4)
        for i, concept in enumerate(filtered_sorted[:40]):
            color = CATEGORY_COLORS.get(concept.category, "#888")
            if cols[i % 4].button(concept.name, key=f"browse_{i}_{concept.name}"):
                query = concept.name

    if query:
        # Entity extraction
        concept_name, confidence = entity_extractor.extract(query)

        if concept_name is None:
            st.error(
                f"Could not identify an AI concept in: **'{query}'**. "
                "Try a more specific term like 'RAG' or 'Transformer'."
            )
            # Show search suggestions
            search_results = client.search_concepts(query, limit=5)
            if search_results:
                st.info("Did you mean one of these?")
                for r in search_results:
                    st.button(r.name, key=f"suggest_{r.name}")
        else:
            concept = client.get_concept(concept_name)
            mode = "standard" if selected_mode == "auto-detect" else selected_mode
            if selected_mode == "auto-detect":
                mode = GraphRetriever.detect_mode(query)

            # ── Concept card ─────────────────────────────────────────────
            col_card, col_graph = st.columns([1, 1.6])

            with col_card:
                diff_color = DIFFICULTY_COLORS.get(concept.difficulty, "#888")
                cat_color = CATEGORY_COLORS.get(concept.category, "#888")

                st.markdown(f"""
                <div class="concept-card">
                    <h3 style="margin:0">{concept.name}</h3>
                    <span style="background:{cat_color}" class="difficulty-badge">{concept.category}</span>
                    <span style="background:{diff_color};margin-left:6px" class="difficulty-badge">{concept.difficulty}</span>
                    <p style="margin-top:12px">{concept.definition}</p>
                </div>
                """, unsafe_allow_html=True)

                if concept.example_use_cases:
                    st.markdown("**Example Use Cases**")
                    for uc in concept.example_use_cases:
                        st.markdown(f"- {uc}")

                if concept.related_tools:
                    st.markdown("**Related Tools**")
                    st.markdown(" ".join([
                        f'<span class="rel-tag">{t}</span>'
                        for t in concept.related_tools
                    ]), unsafe_allow_html=True)

            with col_graph:
                # Show mini subgraph
                subgraph = graph_retriever.retrieve(concept_name, mode="standard", hops=1)
                from dashboard.graph_viz import build_subgraph_html
                html = build_subgraph_html(subgraph, pagerank_scores, height="480px")
                st.components.v1.html(html, height=490, scrolling=False)

            # ── Relationships panel ──────────────────────────────────────
            st.markdown("#### Relationships")
            subgraph_full = graph_retriever.retrieve(concept_name, mode=mode, hops=2)

            rel_by_type: dict[str, list] = {}
            for rel in subgraph_full.relationships:
                if rel.source == concept_name or rel.target == concept_name:
                    rt = rel.rel_type
                    rel_by_type.setdefault(rt, [])
                    other = rel.target if rel.source == concept_name else rel.source
                    direction = "→" if rel.source == concept_name else "←"
                    rel_by_type[rt].append((direction, other))

            cols = st.columns(min(len(rel_by_type), 3) or 1)
            for i, (rt, items) in enumerate(rel_by_type.items()):
                with cols[i % len(cols)]:
                    st.markdown(f"**{rt}**")
                    for direction, other in items:
                        other_concept = client.get_concept(other)
                        preview = other_concept.short_description() if other_concept else ""
                        st.markdown(
                            f'<span class="rel-tag">{direction} {other}</span>',
                            unsafe_allow_html=True,
                        )
                        if preview:
                            st.caption(preview[:80])

            # ── Ask a question ───────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### Ask a question about this concept")
            question = st.text_input(
                "Your question",
                value=query if "?" in query else "",
                key="ask_question",
                placeholder=f"e.g. How does {concept_name} work? What are its use cases?",
            )
            if st.button("Generate Answer", type="primary"):
                with st.spinner("Querying knowledge graph + Ollama..."):
                    vector_results = vector_retriever.search(question, exclude_names=[concept_name])
                    context = assemble_context(subgraph_full, vector_results=vector_results)
                    result = answer_generator.generate(question or query, context, concept_name)

                if result.error:
                    st.warning(f"Note: {result.error}")

                st.markdown("#### Answer")
                st.markdown(result.answer)

                with st.expander("View graph context used"):
                    st.code(result.context_used, language="markdown")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: GRAPH EXPLORER
# ═══════════════════════════════════════════════════════════════════════════
with tab_graph:
    st.header("Knowledge Graph Explorer")
    st.caption(
        "Explore the full AI concept knowledge graph. "
        "Node size = PageRank importance. Color = category. Click nodes to explore."
    )

    col_filter1, col_filter2, col_filter3 = st.columns(3)
    with col_filter1:
        show_categories = st.multiselect(
            "Show categories",
            options=sorted(CATEGORY_COLORS.keys()),
            default=list(CATEGORY_COLORS.keys()),
        )
    with col_filter2:
        show_difficulties = st.multiselect(
            "Show difficulties",
            options=["beginner", "intermediate", "advanced", "expert"],
            default=["beginner", "intermediate", "advanced", "expert"],
        )
    with col_filter3:
        highlight_node = st.selectbox(
            "Highlight concept",
            options=["(none)"] + sorted(client.get_all_concept_names()),
        )

    # Collect all relationships for filtered nodes
    all_concepts = get_all_concepts()
    filtered_nodes = [
        c for c in all_concepts
        if c.category in show_categories and c.difficulty in show_difficulties
    ]
    filtered_names = {c.name for c in filtered_nodes}

    # Gather all edges between filtered nodes
    all_rels = []
    for c in filtered_nodes:
        subgraph = client.get_neighbors(c.name, hops=1)
        for rel in subgraph.relationships:
            if rel.source in filtered_names and rel.target in filtered_names:
                all_rels.append(rel)

    # ── View selector (sunburst first = default) ──────────────────────────────
    viz_tab_sunburst, viz_tab_treemap, viz_tab_net, viz_tab_heatmap = st.tabs([
        "🌞 Sunburst", "▦ Treemap", "🕸 Network Graph", "🔥 Relationship Heatmap"
    ])

    # ── Sunburst ─────────────────────────────────────────────────────────────
    with viz_tab_sunburst:
        st.caption(
            "Inner ring = category · Outer ring = concept · "
            "Sector size = PageRank importance · Hover for definition"
        )
        from dashboard.charts import build_sunburst
        fig_sun = build_sunburst(filtered_nodes, pagerank_scores)
        st.plotly_chart(fig_sun, use_container_width=True)

    # ── Treemap ──────────────────────────────────────────────────────────────
    with viz_tab_treemap:
        st.caption(
            "Nested rectangles grouped by category · "
            "Rectangle size = PageRank importance · Hover for definition"
        )
        from dashboard.charts import build_treemap
        fig_tree = build_treemap(filtered_nodes, pagerank_scores)
        st.plotly_chart(fig_tree, use_container_width=True)

    # ── Network graph ────────────────────────────────────────────────────────
    with viz_tab_net:
        from dashboard.graph_viz import build_full_graph_html
        highlight = highlight_node if highlight_node != "(none)" else None
        html = build_full_graph_html(
            filtered_nodes,
            all_rels,
            pagerank_scores=pagerank_scores,
            height="600px",
            highlight=highlight,
        )
        st.components.v1.html(html, height=620, scrolling=False)

        st.markdown("**Category Legend**")
        legend_cols = st.columns(len(CATEGORY_COLORS))
        for i, (cat, color) in enumerate(CATEGORY_COLORS.items()):
            legend_cols[i % len(legend_cols)].markdown(
                f'<span style="color:{color}">■</span> {cat}',
                unsafe_allow_html=True,
            )

    # ── Relationship Heatmap ─────────────────────────────────────────────────
    with viz_tab_heatmap:
        st.caption(
            "Row = source category · Column = target category · "
            "Color intensity = number of relationships · Hover for breakdown by type"
        )
        from dashboard.charts import build_heatmap
        seen_rels = set()
        unique_rels = []
        for r in all_rels:
            key = (r.source, r.rel_type, r.target)
            if key not in seen_rels:
                seen_rels.add(key)
                unique_rels.append(r)
        fig_heatmap = build_heatmap(filtered_nodes, unique_rels)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # ── PageRank leaderboard ─────────────────────────────────────────────────
    if pagerank_scores:
        st.markdown("---")
        st.markdown("#### Most Central Concepts (by PageRank)")
        top_concepts = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        rank_cols = st.columns(5)
        for i, (name, score) in enumerate(top_concepts):
            c = client.get_concept(name)
            if c:
                color = CATEGORY_COLORS.get(c.category, "#888")
                rank_cols[i % 5].markdown(
                    f'<div style="border-left:3px solid {color};padding-left:8px">'
                    f'<b>{name}</b><br><small>{c.category}</small><br>'
                    f'<small style="color:#888">score: {score:.3f}</small></div>',
                    unsafe_allow_html=True,
                )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: LEARNING PATH
# ═══════════════════════════════════════════════════════════════════════════
with tab_path:
    st.header("Learning Path Generator")
    st.caption(
        "Generate a personalized curriculum to learn any AI concept, "
        "starting from what you already know."
    )

    all_names_sorted = sorted(client.get_all_concept_names())

    col_target, col_known = st.columns([1, 2])
    with col_target:
        target = st.selectbox(
            "I want to learn:",
            options=all_names_sorted,
            index=all_names_sorted.index("GraphRAG") if "GraphRAG" in all_names_sorted else 0,
        )
    with col_known:
        known = st.multiselect(
            "I already know:",
            options=all_names_sorted,
            default=[],
            help="Select concepts you're already comfortable with",
        )

    col_path, col_curriculum = st.columns(2)

    with col_path:
        st.markdown("#### Direct Learning Path")
        if known:
            from_concept = known[-1]
            try:
                from src.graph.networkx_client import NetworkXClient
                if isinstance(client, NetworkXClient):
                    path = find_learning_path(client, from_concept, target)
                else:
                    path = [from_concept, target]
            except Exception:
                path = [from_concept, target]

            if len(path) > 1:
                for i, step in enumerate(path):
                    step_concept = client.get_concept(step)
                    icon = "🎯" if step == target else ("✅" if step in known else f"{i+1}.")
                    color = CATEGORY_COLORS.get(step_concept.category if step_concept else "Concept", "#888")
                    st.markdown(
                        f'<div style="padding:6px;border-left:3px solid {color};margin:4px 0">'
                        f'{icon} <b>{step}</b>'
                        + (f'<br><small style="color:#888">{step_concept.short_description()}</small>' if step_concept else "")
                        + "</div>",
                        unsafe_allow_html=True,
                    )
                    if i < len(path) - 1:
                        st.markdown('<div style="margin-left:16px;color:#666">↓</div>', unsafe_allow_html=True)
            else:
                st.info(f"No prerequisite path found from '{from_concept}' to '{target}'. "
                        f"Try the curriculum view on the right.")
        else:
            st.info("Select at least one concept you already know to see a direct learning path.")

    with col_curriculum:
        st.markdown("#### Full Curriculum")
        if st.button("Generate Curriculum", type="primary"):
            try:
                from src.graph.networkx_client import NetworkXClient
                if isinstance(client, NetworkXClient):
                    curriculum = generate_curriculum(client, target, known)
                else:
                    curriculum = [target]
            except Exception as e:
                curriculum = [target]
                st.warning(str(e))

            if not curriculum:
                st.success(f"🎉 You already know everything needed for **{target}**!")
            else:
                st.markdown(f"To learn **{target}**, study these concepts in order:")
                for i, c_name in enumerate(curriculum):
                    c = client.get_concept(c_name)
                    is_target = c_name == target
                    is_known = c_name in known
                    diff_color = DIFFICULTY_COLORS.get(c.difficulty if c else "intermediate", "#888")

                    prefix = "🎯" if is_target else ("✅" if is_known else f"{i+1}.")
                    bg = "#E8F5EC" if is_known else ("#FFE4E8" if is_target else "#FFF0F3")

                    st.markdown(
                        f'<div style="background:{bg};border-radius:6px;padding:8px;margin:4px 0">'
                        f'<b>{prefix} {c_name}</b>'
                        + (f' <span style="background:{diff_color};padding:1px 6px;border-radius:4px;font-size:11px;color:white">{c.difficulty}</span>' if c else "")
                        + (f'<br><small style="color:#aaa">{c.short_description()}</small>' if c else "")
                        + "</div>",
                        unsafe_allow_html=True,
                    )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: COMPARE CONCEPTS
# ═══════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.header("Compare Concepts")
    st.caption(
        "Compare two AI concepts side-by-side. "
        "See their definitions, shared prerequisites, and unique dependencies."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        concept_a = st.selectbox(
            "Concept A",
            options=all_names_sorted,
            index=all_names_sorted.index("Retrieval-Augmented Generation")
                  if "Retrieval-Augmented Generation" in all_names_sorted else 0,
        )
    with col_b:
        concept_b = st.selectbox(
            "Concept B",
            options=all_names_sorted,
            index=all_names_sorted.index("Fine-tuning")
                  if "Fine-tuning" in all_names_sorted else 1,
        )

    if concept_a == concept_b:
        st.warning("Select two different concepts to compare.")
    else:
        node_a = client.get_concept(concept_a)
        node_b = client.get_concept(concept_b)

        # Side-by-side concept cards
        col_left, col_mid, col_right = st.columns([5, 1, 5])

        for col, node in [(col_left, node_a), (col_right, node_b)]:
            color = CATEGORY_COLORS.get(node.category, "#888")
            diff_color = DIFFICULTY_COLORS.get(node.difficulty, "#888")
            with col:
                st.markdown(f"""
                <div class="concept-card" style="border-left-color:{color}">
                    <h4 style="margin:0">{node.name}</h4>
                    <span style="background:{color}" class="difficulty-badge">{node.category}</span>
                    <span style="background:{diff_color};margin-left:6px" class="difficulty-badge">{node.difficulty}</span>
                    <p style="margin-top:10px;font-size:14px">{node.definition}</p>
                </div>
                """, unsafe_allow_html=True)

                if node.example_use_cases:
                    st.markdown("**Use Cases**")
                    for uc in node.example_use_cases[:3]:
                        st.markdown(f"- {uc}")

        with col_mid:
            st.markdown("<div style='padding-top:60px;text-align:center;font-size:24px'>vs</div>",
                        unsafe_allow_html=True)

        # Prerequisite intersection
        st.markdown("---")
        st.markdown("#### Graph-Based Comparison")

        from src.graph.queries import get_comparison
        comparison = get_comparison(client, concept_a, concept_b)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Unique to {concept_a}**")
            for prereq in comparison["unique_to_a"][:8] or ["(none)"]:
                c = client.get_concept(prereq)
                color = CATEGORY_COLORS.get(c.category if c else "Concept", "#888")
                st.markdown(
                    f'<span style="border-left:3px solid {color};padding-left:6px">{prereq}</span>',
                    unsafe_allow_html=True,
                )
        with col2:
            st.markdown("**Shared Prerequisites**")
            shared = comparison["shared_prerequisites"]
            if shared:
                for prereq in shared[:8]:
                    c = client.get_concept(prereq)
                    color = CATEGORY_COLORS.get(c.category if c else "Concept", "#888")
                    st.markdown(
                        f'<span style="border-left:3px solid {color};padding-left:6px">⚡ {prereq}</span>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No shared prerequisites found.")
        with col3:
            st.markdown(f"**Unique to {concept_b}**")
            for prereq in comparison["unique_to_b"][:8] or ["(none)"]:
                c = client.get_concept(prereq)
                color = CATEGORY_COLORS.get(c.category if c else "Concept", "#888")
                st.markdown(
                    f'<span style="border-left:3px solid {color};padding-left:6px">{prereq}</span>',
                    unsafe_allow_html=True,
                )

        # LLM comparison answer
        if answer_generator.available:
            st.markdown("---")
            st.markdown("#### When to use each?")
            if st.button("Generate Comparison Answer", type="primary"):
                with st.spinner("Generating comparison from graph context..."):
                    subgraph_a = graph_retriever.retrieve(concept_a, mode="standard", hops=2)
                    subgraph_b = graph_retriever.retrieve(concept_b, mode="standard", hops=2)
                    context_a = assemble_context(subgraph_a)
                    context_b = assemble_context(subgraph_b)
                    combined_context = (
                        f"=== {concept_a} ===\n{context_a}\n\n"
                        f"=== {concept_b} ===\n{context_b}"
                    )
                    question = f"Compare {concept_a} and {concept_b}. When should you use each one? What are the key differences?"
                    result = answer_generator.generate(question, combined_context, f"{concept_a} vs {concept_b}")
                st.markdown(result.answer)
