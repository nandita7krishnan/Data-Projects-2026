"""
Global retriever — the map-reduce retrieval engine for GraphRAG.

For broad, thematic queries that span multiple concept communities:
  1. RANK: score all community summaries against the query
  2. MAP: ask Ollama to partially answer the query using each top community
  3. REDUCE: synthesize partial answers into a final coherent response

This is the key addition that makes this proper GraphRAG (vs. the existing
local BFS retrieval which handles specific concept queries).

Usage:
    from src.retrieval.global_retriever import GlobalRetriever
    retriever = GlobalRetriever(communities, ollama_client, model)
    result = retriever.retrieve("What are the main approaches to language modeling?")
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from config.settings import GLOBAL_MAX_COMMUNITY_TOKENS, GLOBAL_TOP_K_COMMUNITIES, OLLAMA_MODEL
from src.graph.schema import Community

logger = logging.getLogger(__name__)

_MAP_PROMPT = """\
You are helping answer a user's question about AI/ML concepts.

Here is a summary of a related community of concepts:

{community_summary}

User question: {query}

If this community summary is relevant to the question, provide a brief (2-3 sentence) \
partial answer based only on the summary above.
If this community is NOT relevant, respond with exactly: IRRELEVANT

Partial answer:"""

_REDUCE_SYSTEM = """\
You are an AI knowledge assistant. You will receive several partial answers \
about AI/ML concepts, each derived from a different part of a knowledge graph. \
Synthesize them into a single, coherent, well-structured answer.

Rules:
- Combine information from all relevant partial answers
- Remove redundancy
- Structure with bullet points if listing multiple items
- Be concise and accurate
- If partial answers conflict, note both perspectives"""

_REDUCE_USER = """\
User question: {query}

Partial answers from different knowledge communities:

{partial_answers}

Synthesize these into a final, comprehensive answer:"""


@dataclass
class GlobalSearchResult:
    """Result of a global map-reduce retrieval."""
    query: str
    final_answer: str
    communities_used: int
    partial_answers: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def success(self) -> bool:
        return not bool(self.error)


def _score_community(summary: str, query: str) -> float:
    """
    Simple TF-IDF-style relevance score between a query and a community summary.
    No external dependencies needed.
    """
    if not summary:
        return 0.0

    query_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
    summary_words = re.findall(r"\b\w{3,}\b", summary.lower())
    summary_word_set = set(summary_words)

    if not query_words or not summary_words:
        return 0.0

    # Term overlap score
    overlap = len(query_words & summary_word_set)
    # Normalize by query length so short queries don't dominate
    score = overlap / len(query_words)

    # Bonus: exact phrase matches (bigrams)
    query_bigrams = set()
    qw = list(query_words)
    for i in range(len(qw) - 1):
        query_bigrams.add(f"{qw[i]} {qw[i+1]}")

    summary_text = summary.lower()
    for bigram in query_bigrams:
        if bigram in summary_text:
            score += 0.2

    return score


def _truncate_summary(summary: str, max_words: int = GLOBAL_MAX_COMMUNITY_TOKENS) -> str:
    """Truncate summary to max_words words to stay within context budget."""
    words = summary.split()
    if len(words) <= max_words:
        return summary
    return " ".join(words[:max_words]) + "..."


class GlobalRetriever:
    """
    Map-reduce retrieval over community summaries.

    Answers broad/thematic questions by:
      1. Ranking communities by relevance to the query
      2. Getting partial answers from top K communities (map step)
      3. Synthesizing into a final answer (reduce step)
    """

    def __init__(
        self,
        communities: list[Community],
        ollama_client,
        model: str = OLLAMA_MODEL,
    ):
        self._communities = [c for c in communities if c.summary]
        self._ollama = ollama_client
        self._model = model

        if not self._communities:
            logger.warning(
                "GlobalRetriever: no communities with summaries loaded. "
                "Run `python main.py ingest` first."
            )

    @property
    def available(self) -> bool:
        return bool(self._communities) and self._ollama is not None

    def retrieve(
        self,
        query: str,
        top_k: int = GLOBAL_TOP_K_COMMUNITIES,
    ) -> GlobalSearchResult:
        """
        Run map-reduce retrieval for a broad query.

        Args:
            query: user's natural language question
            top_k: number of top communities to map over

        Returns:
            GlobalSearchResult with final_answer and metadata
        """
        if not self._communities:
            return GlobalSearchResult(
                query=query,
                final_answer="No community summaries available. Run `python main.py ingest` first.",
                communities_used=0,
                error="No communities loaded.",
            )

        if self._ollama is None:
            return GlobalSearchResult(
                query=query,
                final_answer=self._fallback_answer(query, top_k),
                communities_used=0,
                error="Ollama not available.",
            )

        # Step 1: rank communities by query relevance
        scored = [
            (c, _score_community(c.summary, query))
            for c in self._communities
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_communities = [c for c, score in scored[:top_k] if score > 0]

        if not top_communities:
            # All zero scores — just take top_k by index as fallback
            top_communities = self._communities[:top_k]

        logger.info(
            "Global retrieval: mapping over %d communities for query: %s",
            len(top_communities), query[:60],
        )

        # Step 2: MAP — partial answer per community
        partial_answers: list[str] = []
        for i, community in enumerate(top_communities):
            summary = _truncate_summary(community.summary)
            prompt = _MAP_PROMPT.format(community_summary=summary, query=query)

            try:
                response = self._ollama.chat(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.2, "num_predict": 150},
                )
                answer = response["message"]["content"].strip()

                if answer.upper() != "IRRELEVANT" and len(answer) > 20:
                    partial_answers.append(answer)
                    logger.debug("Community %d: got partial answer.", community.community_id)
                else:
                    logger.debug("Community %d: marked irrelevant.", community.community_id)

            except Exception as e:
                logger.warning("Map step failed for community %d: %s", community.community_id, e)

        if not partial_answers:
            return GlobalSearchResult(
                query=query,
                final_answer=self._fallback_answer(query, top_k),
                communities_used=0,
                error="No relevant communities found for this query.",
            )

        # Step 3: REDUCE — synthesize partial answers
        numbered = "\n\n".join(
            f"[{i+1}] {ans}" for i, ans in enumerate(partial_answers)
        )
        reduce_user = _REDUCE_USER.format(query=query, partial_answers=numbered)

        try:
            response = self._ollama.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": _REDUCE_SYSTEM},
                    {"role": "user", "content": reduce_user},
                ],
                options={"temperature": 0.3, "num_predict": 600},
            )
            final_answer = response["message"]["content"].strip()

        except Exception as e:
            logger.error("Reduce step failed: %s", e)
            final_answer = "\n\n".join(partial_answers)

        return GlobalSearchResult(
            query=query,
            final_answer=final_answer,
            communities_used=len(partial_answers),
            partial_answers=partial_answers,
        )

    def _fallback_answer(self, query: str, top_k: int) -> str:
        """Return community summaries directly when Ollama is unavailable."""
        scored = sorted(
            self._communities,
            key=lambda c: _score_community(c.summary, query),
            reverse=True,
        )[:top_k]

        parts = [f"**Community Summaries (LLM unavailable)**\n"]
        for c in scored:
            if c.summary:
                parts.append(f"- {c.summary}")
        return "\n".join(parts)
