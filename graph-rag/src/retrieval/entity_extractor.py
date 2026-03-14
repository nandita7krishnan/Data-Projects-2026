"""
Entity extractor — identifies the AI concept being asked about in a query.

Two-stage strategy:
  1. Fast path: fuzzy substring matching against all known concept names
  2. LLM fallback: Ollama call when confidence is low
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from config.settings import ENTITY_CONFIDENCE_THRESHOLD, OLLAMA_MODEL

logger = logging.getLogger(__name__)

_ENTITY_PROMPT = """\
You are an AI terminology expert.

Given the following question about AI/ML, identify the single most relevant \
AI/ML concept being asked about.

Question: {query}

Known concepts: {concept_list}

Respond with ONLY valid JSON:
{{"concept": "exact concept name from the list above, or null if none match"}}
"""


class EntityExtractor:
    """
    Extracts the primary AI concept from a natural language query.

    Usage:
        extractor = EntityExtractor(concept_names, ollama_client, model)
        concept_name, confidence = extractor.extract("What is RAG?")
    """

    def __init__(
        self,
        concept_names: list[str],
        ollama_client=None,
        model: str = OLLAMA_MODEL,
    ):
        self._names = concept_names
        self._names_lower = {n.lower(): n for n in concept_names}
        # Build acronym map: "Retrieval-Augmented Generation" → "rag"
        self._acronym_map = _build_acronym_map(concept_names)
        self._ollama = ollama_client
        self._model = model

    def refresh_names(self, concept_names: list[str]) -> None:
        """Update the known concept set (call after graph is updated)."""
        self._names = concept_names
        self._names_lower = {n.lower(): n for n in concept_names}
        self._acronym_map = _build_acronym_map(concept_names)

    def extract(self, query: str) -> tuple[Optional[str], float]:
        """
        Returns (concept_name, confidence) or (None, 0.0) if not found.
        confidence is in [0.0, 1.0].
        """
        # Stage 1: keyword matching
        result, confidence = self._keyword_match(query)
        if confidence >= ENTITY_CONFIDENCE_THRESHOLD:
            logger.debug("Entity extracted by keyword: '%s' (%.2f)", result, confidence)
            return result, confidence

        # Stage 2: LLM fallback
        if self._ollama is not None:
            result_llm, conf_llm = self._llm_extract(query)
            if result_llm and conf_llm > confidence:
                logger.debug("Entity extracted by LLM: '%s'", result_llm)
                return result_llm, conf_llm

        return result, confidence

    # ── Stage 1: keyword matching ────────────────────────────────────────────

    def _keyword_match(self, query: str) -> tuple[Optional[str], float]:
        """Score-based substring and token matching, including acronym resolution."""
        q_lower = query.lower()
        best_name: Optional[str] = None
        best_score = 0.0

        # Check each word in the query against the acronym map
        for word in re.split(r"\W+", q_lower):
            if word in self._acronym_map:
                return self._acronym_map[word], 0.92

        for name_lower, canonical in self._names_lower.items():
            score = 0.0

            # Exact match — highest confidence
            if q_lower == name_lower:
                return canonical, 1.0

            # Full name appears in query
            if name_lower in q_lower:
                # Longer match → higher confidence (avoid short spurious matches)
                score = min(0.95, 0.7 + len(name_lower) / 100)

            # Query words appear in the concept name
            query_words = set(re.split(r"\W+", q_lower)) - {"what", "is", "a", "the",
                                                              "how", "does", "do", "are",
                                                              "explain", "define", "tell",
                                                              "me", "about", "and", "or"}
            name_words = set(re.split(r"\W+", name_lower))
            overlap = query_words & name_words
            if overlap:
                word_score = len(overlap) / max(len(name_words), 1) * 0.8
                score = max(score, word_score)

            if score > best_score:
                best_score = score
                best_name = canonical

        return best_name, best_score

    # ── Stage 2: LLM fallback ───────────────────────────────────────────────

    def _llm_extract(self, query: str) -> tuple[Optional[str], float]:
        """Use Ollama to identify the concept when keyword matching fails."""
        # Show top 30 most relevant concept names to keep prompt short
        relevant_names = self._find_related_names(query, limit=30)
        concept_list = ", ".join(relevant_names) if relevant_names else ", ".join(self._names[:30])

        prompt = _ENTITY_PROMPT.format(query=query, concept_list=concept_list)

        try:
            response = self._ollama.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
            )
            content = response["message"]["content"]
            json_match = re.search(r"\{.*?\}", content, re.DOTALL)
            if not json_match:
                return None, 0.0
            parsed = json.loads(json_match.group())
            concept = parsed.get("concept")
            if concept and concept.lower() in self._names_lower:
                return self._names_lower[concept.lower()], 0.85
            return None, 0.0
        except Exception as e:
            logger.warning("LLM entity extraction failed: %s", e)
            return None, 0.0

    def _find_related_names(self, query: str, limit: int = 30) -> list[str]:
        """Quick shortlist of concept names that share words with the query."""
        q_words = set(re.split(r"\W+", query.lower()))
        scored = []
        for name_lower, canonical in self._names_lower.items():
            name_words = set(re.split(r"\W+", name_lower))
            overlap = len(q_words & name_words)
            if overlap > 0:
                scored.append((overlap, canonical))
        scored.sort(reverse=True)
        return [n for _, n in scored[:limit]] or list(self._names[:limit])


def _build_acronym_map(concept_names: list[str]) -> dict[str, str]:
    """
    Build a map from lowercase acronym → canonical concept name.

    "Retrieval-Augmented Generation" → "rag"
    "Large Language Model"           → "llm"
    "Key-Value Cache"                → "kv cache" (skipped, not single token)
    """
    acronym_map: dict[str, str] = {}
    for name in concept_names:
        words = re.split(r"[\s\-_]+", name)
        if len(words) >= 2:
            acronym = "".join(w[0] for w in words if w).lower()
            if len(acronym) >= 2 and acronym not in acronym_map:
                acronym_map[acronym] = name
    return acronym_map
