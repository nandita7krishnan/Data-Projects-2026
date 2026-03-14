"""
Answer generator — produces natural language answers from assembled graph context.

Uses Ollama for local, free inference. The LLM receives:
  - A system prompt anchoring it to the graph context
  - The assembled context (structured graph subgraph + optional vector results)
  - The original user query

This is the final step in the Graph RAG pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an AI knowledge assistant specializing in explaining AI/ML concepts.

You will be given structured context extracted from a knowledge graph about AI concepts.
The context includes:
- The main concept definition and metadata
- Direct relationships to other concepts (e.g., REQUIRES, IS_A, BUILT_WITH)
- Related concepts discovered via graph traversal
- Optional supporting text from semantic search

Your task:
1. Answer the user's question clearly and accurately
2. Ground your answer ENTIRELY in the provided context — do not make up information
3. Explain concepts in plain English, accessible to someone new to AI
4. When relevant, mention prerequisites, related concepts, and real-world use cases
5. If the context doesn't contain enough information, say so honestly

Keep your answer concise and well-structured. Use bullet points or numbered lists \
when listing multiple items.
"""

_USER_TEMPLATE = """\
=== KNOWLEDGE GRAPH CONTEXT ===
{context}

=== USER QUESTION ===
{query}

Please answer the question based on the context above.
"""


@dataclass
class AnswerResult:
    """Structured result from the answer generator."""
    query: str
    concept: str
    answer: str
    context_used: str
    sources: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def success(self) -> bool:
        return not bool(self.error)


class AnswerGenerator:
    """
    Generates grounded natural language answers using Ollama.

    Usage:
        generator = AnswerGenerator()
        result = generator.generate(query, context, concept_name)
    """

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ):
        self._model = model
        self._base_url = base_url
        self._client = self._init_client()

    def _init_client(self):
        try:
            import ollama
            client = ollama.Client(host=self._base_url)
            # Test connectivity
            client.list()
            logger.info("Ollama client connected at %s, model: %s", self._base_url, self._model)
            return client
        except ImportError:
            logger.warning("ollama package not installed. LLM generation disabled.")
            return None
        except Exception as e:
            logger.warning(
                "Ollama not available at %s (%s). LLM generation disabled.",
                self._base_url, e,
            )
            return None

    @property
    def available(self) -> bool:
        return self._client is not None

    def get_client(self):
        """Expose the Ollama client for other components (e.g., entity extractor)."""
        return self._client

    def generate(
        self,
        query: str,
        context: str,
        concept_name: str,
        sources: list[str] = None,
    ) -> AnswerResult:
        """
        Generate a grounded answer for `query` using the assembled `context`.

        Falls back to returning the raw context as the answer if Ollama is unavailable.
        """
        if self._client is None:
            return AnswerResult(
                query=query,
                concept=concept_name,
                answer=self._fallback_answer(context, concept_name),
                context_used=context,
                sources=sources or [],
                error="Ollama not available — showing raw graph context.",
            )

        user_message = _USER_TEMPLATE.format(context=context, query=query)

        try:
            response = self._client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                options={
                    "temperature": 0.3,
                    "num_predict": 800,
                },
            )
            answer = response["message"]["content"].strip()
            return AnswerResult(
                query=query,
                concept=concept_name,
                answer=answer,
                context_used=context,
                sources=sources or [concept_name],
            )

        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            return AnswerResult(
                query=query,
                concept=concept_name,
                answer=self._fallback_answer(context, concept_name),
                context_used=context,
                sources=sources or [],
                error=str(e),
            )

    def _fallback_answer(self, context: str, concept_name: str) -> str:
        """Structured fallback when LLM is unavailable."""
        return (
            f"**{concept_name}** (graph context — LLM unavailable)\n\n"
            f"{context}"
        )
