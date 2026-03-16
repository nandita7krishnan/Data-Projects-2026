"""
Community summarizer — generates LLM summaries for each detected community.

This is one of the two core GraphRAG additions (along with global_retriever).
Each community cluster of concepts gets a paragraph-length summary written
by the LLM, which is then used during global retrieval.

Usage:
    from src.ingestion.community_summarizer import generate_community_summaries
    communities = generate_community_summaries(communities, graph_client, ollama_client, model)
"""
from __future__ import annotations

import json
import logging

from config.settings import COMMUNITY_SUMMARIES_PATH
from src.graph.schema import Community

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = """\
You are building a knowledge retrieval system about AI and machine learning concepts.

The following concepts form a closely related community in a knowledge graph:

Concepts: {concept_names}

Concept details:
{concept_details}

Write a 2-3 sentence summary that:
1. Names the central theme or subject area of this community
2. Explains how these concepts relate to each other
3. Notes the key practical significance or applications

Be specific and factual. Do not use filler phrases like "This community covers...".
Start directly with the theme.

Summary:"""


def _build_concept_details(concept_names: list[str], graph_client, top_n: int = 10) -> str:
    """Fetch concept definitions from the graph, limited to top_n by PageRank."""
    try:
        from src.graph.algorithms import get_concept_importance
        from src.graph.networkx_client import NetworkXClient
        if isinstance(graph_client, NetworkXClient):
            scores = get_concept_importance(graph_client)
            # Sort concept_names by PageRank score, take top_n
            ranked = sorted(concept_names, key=lambda n: scores.get(n, 0), reverse=True)
            concept_names = ranked[:top_n]
    except Exception:
        concept_names = concept_names[:top_n]

    lines = []
    for name in concept_names:
        concept = graph_client.get_concept(name)
        if concept:
            lines.append(f"- {name}: {concept.definition}")
        else:
            lines.append(f"- {name}")
    return "\n".join(lines)


def generate_community_summaries(
    communities: list[Community],
    graph_client,
    ollama_client,
    model: str,
) -> list[Community]:
    """
    For each community, call Ollama to generate a summary paragraph.
    Saves results to COMMUNITY_SUMMARIES_PATH after each community (resumable).

    Args:
        communities: list of Community objects (from detect_communities)
        graph_client: graph backend for fetching concept definitions
        ollama_client: connected Ollama client
        model: Ollama model name

    Returns:
        communities with .summary populated
    """
    # Load any already-summarized communities (resumability)
    existing: dict[int, str] = {}
    if COMMUNITY_SUMMARIES_PATH.exists():
        try:
            with open(COMMUNITY_SUMMARIES_PATH, encoding="utf-8") as f:
                saved = json.load(f)
            existing = {d["community_id"]: d.get("summary", "") for d in saved}
        except Exception:
            existing = {}

    total = len(communities)
    for i, community in enumerate(communities):
        # Resume: skip if already summarized
        if community.community_id in existing and existing[community.community_id]:
            community.summary = existing[community.community_id]
            logger.debug("Community %d: loaded cached summary.", community.community_id)
            continue

        print(
            f"  [{i+1}/{total}] Summarizing community {community.community_id} "
            f"({len(community.node_names)} concepts)...",
            end="\r",
        )

        concept_details = _build_concept_details(community.node_names, graph_client)
        prompt = _SUMMARY_PROMPT.format(
            concept_names=", ".join(community.node_names),
            concept_details=concept_details,
        )

        try:
            response = ollama_client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2, "num_predict": 200},
            )
            summary = response["message"]["content"].strip()
        except Exception as e:
            logger.warning("Community %d summary failed: %s", community.community_id, e)
            summary = f"Community of {len(community.node_names)} AI concepts: {', '.join(community.node_names[:5])}."

        community.summary = summary

        # Persist after each community (resumable)
        _save_summaries(communities)

    print()  # clear \r line
    logger.info("Generated summaries for %d communities.", total)
    return communities


def load_community_summaries(path=COMMUNITY_SUMMARIES_PATH) -> list[Community]:
    """Load communities with summaries from the persisted JSON file."""
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return [Community.from_dict(d) for d in data]
    except Exception as e:
        logger.warning("Could not load community summaries: %s", e)
        return []


def _save_summaries(communities: list[Community]) -> None:
    """Persist communities (with summaries) to JSON."""
    COMMUNITY_SUMMARIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(COMMUNITY_SUMMARIES_PATH, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in communities], f, indent=2)
