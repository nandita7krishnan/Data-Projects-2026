"""
Query router — decides whether a query should go to the global (map-reduce)
or local (BFS graph traversal) retrieval path.

Global path: broad, thematic questions spanning multiple concepts
  e.g. "What are the main approaches to language modeling?"
       "How has attention changed modern NLP?"
       "Give me an overview of training techniques"

Local path: specific concept lookups, comparisons, prerequisites
  e.g. "What is RAG?"
       "What does Transformer require?"
       "Compare FAISS and ChromaDB"

Routing uses a two-stage strategy:
  1. Fast heuristic keyword matching (no LLM call)
  2. LLM fallback for ambiguous queries (small, fast prompt)

Usage:
    from src.retrieval.query_router import route_query
    path = route_query("What are the main trends in AI?")  # → "global"
    path = route_query("What is a Transformer?")            # → "local"
"""
from __future__ import annotations

import logging
import re

from config.settings import OLLAMA_MODEL

logger = logging.getLogger(__name__)

# Keywords strongly indicating a GLOBAL (thematic/overview) query
_GLOBAL_SIGNALS = [
    r"\boverview\b",
    r"\bsummariz",
    r"\blandscape\b",
    r"\btrend",
    r"\bmain (approach|method|technique|way|type)",
    r"\bdifferent (approach|method|technique|type|kind)",
    r"\ball (approach|method|technique|way|type|concept)",
    r"\bhow has\b",
    r"\bhow have\b",
    r"\bwhat are the (main|key|major|different|various|common)",
    r"\bbroadly\b",
    r"\bin general\b",
    r"\bfield of\b",
    r"\bacross (the|different|various)\b",
    r"\bstate of the art\b",
    r"\bevolution of\b",
    r"\bhistory of\b",
    r"\blandscape of\b",
    r"\bclassif(y|ication) of\b",
    r"\btaxonom",
    r"\bcategor",
]

# Keywords strongly indicating a LOCAL (specific concept) query
_LOCAL_SIGNALS = [
    r"\bwhat is (a |an |the )?\w",
    r"\bdefine\b",
    r"\bexplain\b",
    r"\bhow does .+ work\b",
    r"\bwhat does .+ (do|mean|require|need)\b",
    r"\brequire",
    r"\bprerequisite",
    r"\bdepend",
    r"\bcompare\b",
    r"\bversus\b",
    r"\bvs\.?\b",
    r"\bdifference between\b",
    r"\blearning path\b",
    r"\bhow to (learn|use|implement|build|train)\b",
    r"\bexample of\b",
    r"\buse case",
    r"\bapplication of\b",
]

_ROUTER_PROMPT = """\
Classify this question as either "global" or "local":

- "global": broad, thematic, overview questions spanning many concepts \
(e.g. "What are the main approaches to X?", "How has X evolved?", "Overview of X field")
- "local": specific concept lookups, definitions, comparisons, prerequisites \
(e.g. "What is X?", "How does X work?", "What does X require?", "Compare X and Y")

Question: {query}

Respond with exactly one word: global or local"""


def route_query(
    query: str,
    ollama_client=None,
    model: str = OLLAMA_MODEL,
) -> str:
    """
    Route a query to "global" or "local" retrieval path.

    Args:
        query: user's natural language question
        ollama_client: optional Ollama client for LLM fallback
        model: Ollama model name

    Returns:
        "global" or "local"
    """
    q = query.lower().strip()

    # Count heuristic signal matches
    global_hits = sum(1 for pattern in _GLOBAL_SIGNALS if re.search(pattern, q))
    local_hits = sum(1 for pattern in _LOCAL_SIGNALS if re.search(pattern, q))

    logger.debug(
        "Router heuristics: global_hits=%d, local_hits=%d for: %s",
        global_hits, local_hits, query[:60],
    )

    # Clear heuristic winner
    if global_hits > local_hits and global_hits >= 1:
        logger.info("Router → global (heuristic, hits=%d)", global_hits)
        return "global"

    if local_hits > global_hits and local_hits >= 1:
        logger.info("Router → local (heuristic, hits=%d)", local_hits)
        return "local"

    # Ambiguous — fall back to LLM classification
    if ollama_client is not None:
        try:
            prompt = _ROUTER_PROMPT.format(query=query)
            response = ollama_client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_predict": 5},
            )
            decision = response["message"]["content"].strip().lower()
            if "global" in decision:
                logger.info("Router → global (LLM)")
                return "global"
            else:
                logger.info("Router → local (LLM)")
                return "local"
        except Exception as e:
            logger.warning("Router LLM fallback failed: %s. Defaulting to local.", e)

    # Default: local (safer, more specific)
    logger.info("Router → local (default)")
    return "local"
