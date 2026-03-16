"""
Corpus fetcher — automatically downloads AI concept documents from:
  1. Wikipedia REST API  (free, no key needed)
  2. arXiv API           (free, no key needed)

Saves raw text files to data/corpus/ so the document loader can chunk them.

Usage:
    from src.ingestion.corpus_fetcher import fetch_corpus
    fetch_corpus(concept_names, corpus_dir)
"""
from __future__ import annotations

import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# arXiv search maps concept names to better search terms
ARXIV_QUERY_OVERRIDES: dict[str, str] = {
    "RAG": "retrieval augmented generation",
    "GraphRAG": "graph retrieval augmented generation",
    "LLM": "large language models",
    "RLHF": "reinforcement learning human feedback",
    "LoRA": "low rank adaptation large language models",
    "FAISS": "FAISS billion scale similarity search",
    "MoE": "mixture of experts language models",
    "PPO": "proximal policy optimization",
    "DPO": "direct preference optimization",
    "CoT": "chain of thought prompting",
    "ReAct": "ReAct synergizing reasoning acting language models",
    # 2026 terms
    "Agentic AI": "autonomous AI agents planning acting",
    "Multi-Agent Orchestration": "multi-agent systems LLM orchestration",
    "AI Assistant": "AI assistant large language model",
    "Multimodal AI": "multimodal large language models",
    "Small Language Model": "small language models efficient inference",
    "Synthetic Data": "synthetic data generation language models",
    "Model Context Protocol": "model context protocol tool use agents",
    "Quantization": "quantization large language models inference",
    "Explainable AI": "explainable artificial intelligence XAI",
    "AI Guardrails": "AI safety guardrails language model alignment",
    "Red-Teaming": "red teaming language models safety evaluation",
    "Physical AI": "embodied AI robotics physical world",
    "Digital Provenance": "digital content provenance deepfake detection watermarking",
    "AI Coding Agent": "AI coding agents software engineering automation",
    "Agent Hooks": "AI agent lifecycle hooks tool use events",
    "Agent Skills": "AI agent skills plugins capabilities",
    "Slash Commands": "AI agent commands prompts automation",
    "MCP Server": "model context protocol server tool integration",
}

# Wikipedia page title overrides — maps concept names to the correct Wikipedia article title
WIKIPEDIA_TITLE_OVERRIDES: dict[str, str] = {
    # existing concepts with mismatched titles
    "Retrieval-Augmented Generation": "Retrieval-augmented generation",
    "Large Language Model": "Large language model",
    "RLHF": "Reinforcement learning from human feedback",
    "LoRA": "Fine-tuning (deep learning)",
    "Sparse Retrieval": "Okapi BM25",
    "Dense Retrieval": "Semantic search",
    "In-context Learning": "Prompt engineering",
    "KV Cache": "Transformer (deep learning architecture)",
    # 2026 terms
    "Explainable AI": "Explainable artificial intelligence",
    "Multimodal AI": "Multimodal learning",
    "Synthetic Data": "Synthetic data",
    "Quantization": "Quantization (signal processing)",
    "Red-Teaming": "Red team",
    "Physical AI": "Embodied cognition",
    "Digital Provenance": "Provenance",
    "AI Assistant": "Virtual assistant",
    "Small Language Model": "Large language model",
    "Agentic AI": "Intelligent agent",
    "Multi-Agent Orchestration": "Multi-agent system",
    "Video Understanding": "Computer vision",
    "AI-Native Development Platform": "Platform as a service",
    "Cursor": "Cursor (text editor)",
    "Agent Skills": "Plugin (computing)",
    "Slash Commands": "Slash (command)",
    "MCP Server": "Client–server model",
    # concepts with no useful Wikipedia page — skip Wikipedia, use other sources
    "CLAUDE.md": None,
    "AGENTS.md": None,
    "Agent Rules": None,
    "Agent Hooks": None,
    "Claude Code": None,
    "AI Coding Agent": None,
    "Model Context Protocol": None,
    "AI Guardrails": None,
    "Slop": None,
}

# Official documentation URLs for concepts that have no Wikipedia article.
# These are fetched as plain text and saved as {name}_docs.txt.
OFFICIAL_DOCS_URLS: dict[str, str] = {
    "Claude Code":             "https://docs.anthropic.com/en/docs/claude-code/overview",
    "CLAUDE.md":               "https://docs.anthropic.com/en/docs/claude-code/memory",
    "Model Context Protocol":  "https://modelcontextprotocol.io/introduction",
    "MCP Server":              "https://modelcontextprotocol.io/docs/concepts/servers",
    "Agent Hooks":             "https://docs.anthropic.com/en/docs/claude-code/hooks",
    "Agent Skills":            "https://docs.anthropic.com/en/docs/claude-code/customization",
    "Slash Commands":          "https://docs.anthropic.com/en/docs/claude-code/slash-commands",
    "Agent Rules":             "https://docs.anthropic.com/en/docs/claude-code/settings",
    "AGENTS.md":               "https://docs.anthropic.com/en/docs/claude-code/memory",
    "AI Guardrails":           "https://github.com/NVIDIA/NeMo-Guardrails/blob/main/README.md",
    "AI Coding Agent":         "https://docs.anthropic.com/en/docs/claude-code/overview",
    "Cursor":                  "https://docs.cursor.com/get-started/introduction",
    "Agentic AI":              "https://www.anthropic.com/research/building-effective-agents",
}


def _safe_filename(name: str) -> str:
    """Convert a concept name to a safe filename."""
    return re.sub(r"[^\w\-]", "_", name).strip("_")


def _fetch_wikipedia(concept_name: str, corpus_dir: Path) -> Optional[Path]:
    """
    Fetch the full Wikipedia article for a concept and save as {name}_wiki.txt.
    Uses the Wikipedia REST API (no key required).
    Respects WIKIPEDIA_TITLE_OVERRIDES — concepts mapped to None are skipped.
    """
    out_path = corpus_dir / f"{_safe_filename(concept_name)}_wiki.txt"
    if out_path.exists():
        logger.debug("Wikipedia already fetched for '%s', skipping.", concept_name)
        return out_path

    # Skip names that Wikipedia can't handle (slashes, very short acronyms, etc.)
    if "/" in concept_name or len(concept_name) <= 2:
        logger.debug("Wikipedia: skipping '%s' (unsupported name format).", concept_name)
        return None

    # Check override table: None means "no Wikipedia page, skip entirely"
    wiki_title = WIKIPEDIA_TITLE_OVERRIDES.get(concept_name, concept_name)
    if wiki_title is None:
        logger.debug("Wikipedia: skipping '%s' (no Wikipedia page).", concept_name)
        return None

    encoded = urllib.parse.quote(wiki_title.replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "GraphRAG-Bot/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            import json
            data = json.loads(resp.read().decode())

        title = data.get("title", concept_name)
        extract = data.get("extract", "")

        if not extract or len(extract) < 50:
            logger.debug("Wikipedia: short/no extract for '%s'.", concept_name)
            return None

        text = f"# {title}\n\nSource: Wikipedia\n\n{extract}\n"
        out_path.write_text(text, encoding="utf-8")
        logger.info("Wikipedia: saved %d chars for '%s'", len(text), concept_name)
        return out_path

    except urllib.error.HTTPError as e:
        if e.code == 404:
            logger.debug("Wikipedia: no article found for '%s'.", concept_name)
        else:
            logger.warning("Wikipedia HTTP %s for '%s': %s", e.code, concept_name, e)
        return None
    except Exception as e:
        logger.warning("Wikipedia fetch failed for '%s': %s", concept_name, e)
        return None


class _ArxivRateLimited(Exception):
    """Raised to signal the caller to stop all arXiv fetching this session."""


def _fetch_arxiv(
    concept_name: str,
    corpus_dir: Path,
    max_results: int = 3,
    max_retries: int = 2,
) -> Optional[Path]:
    """
    Fetch top arXiv abstracts for a concept and save as {name}_arxiv.txt.
    Uses the arXiv API (no key required).

    Raises _ArxivRateLimited if arXiv persistently returns 429, so the
    caller can abort remaining arXiv calls for this session.
    """
    out_path = corpus_dir / f"{_safe_filename(concept_name)}_arxiv.txt"
    if out_path.exists():
        logger.debug("arXiv already fetched for '%s', skipping.", concept_name)
        return out_path

    search_term = ARXIV_QUERY_OVERRIDES.get(concept_name, concept_name)
    encoded_query = urllib.parse.quote(f'ti:"{search_term}" OR abs:"{search_term}"')
    url = (
        f"http://export.arxiv.org/api/query"
        f"?search_query={encoded_query}"
        f"&max_results={max_results}"
        f"&sortBy=relevance"
    )

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "GraphRAG-Bot/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                content = resp.read().decode("utf-8")

            titles = re.findall(r"<title>(.*?)</title>", content, re.DOTALL)
            summaries = re.findall(r"<summary>(.*?)</summary>", content, re.DOTALL)
            titles = titles[1:]  # skip feed title

            if not summaries:
                logger.debug("arXiv: no results for '%s'.", concept_name)
                return None

            parts = [f"# arXiv Papers: {concept_name}\n\nSource: arXiv\n"]
            for i, (title, summary) in enumerate(zip(titles, summaries)):
                title = re.sub(r"\s+", " ", title).strip()
                summary = re.sub(r"\s+", " ", summary).strip()
                parts.append(f"\n## Paper {i+1}: {title}\n\n{summary}\n")

            text = "\n".join(parts)
            out_path.write_text(text, encoding="utf-8")
            logger.info("arXiv: saved %d chars for '%s'", len(text), concept_name)
            return out_path

        except urllib.error.HTTPError as e:
            if e.code == 429:
                if attempt < max_retries - 1:
                    wait = 15 * (2 ** attempt)  # 15s, 30s
                    logger.warning("arXiv 429 for '%s'. Waiting %ds...", concept_name, wait)
                    time.sleep(wait)
                else:
                    # Persistent rate limit — signal caller to stop arXiv for this session
                    raise _ArxivRateLimited(concept_name)
            else:
                logger.warning("arXiv HTTP %s for '%s': %s", e.code, concept_name, e)
                return None
        except _ArxivRateLimited:
            raise
        except Exception as e:
            logger.warning("arXiv fetch failed for '%s': %s", concept_name, e)
            return None

    return None


def _fetch_official_docs(concept_name: str, corpus_dir: Path) -> Optional[Path]:
    """
    Fetch an official documentation page for a concept and save as {name}_docs.txt.
    Only attempted for concepts listed in OFFICIAL_DOCS_URLS.
    """
    url = OFFICIAL_DOCS_URLS.get(concept_name)
    if url is None:
        return None

    out_path = corpus_dir / f"{_safe_filename(concept_name)}_docs.txt"
    if out_path.exists():
        logger.debug("Docs already fetched for '%s', skipping.", concept_name)
        return out_path

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "GraphRAG-Bot/1.0",
                "Accept": "text/html,text/plain",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8", errors="replace")

        # Strip HTML tags to get plain text
        text = re.sub(r"<script[^>]*>.*?</script>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&[a-z]+;", " ", text)
        text = re.sub(r"\s{3,}", "\n\n", text).strip()

        if len(text) < 100:
            logger.debug("Docs: too short for '%s' from %s", concept_name, url)
            return None

        header = f"# {concept_name}\n\nSource: {url}\n\n"
        out_path.write_text(header + text[:8000], encoding="utf-8")  # cap at 8k chars
        logger.info("Docs: saved %d chars for '%s'", len(text), concept_name)
        return out_path

    except Exception as e:
        logger.warning("Docs fetch failed for '%s' (%s): %s", concept_name, url, e)
        return None


def _write_seed_fallback(concept_name: str, concept_data: dict, corpus_dir: Path) -> Optional[Path]:
    """
    Write a corpus file from the seed definition for concepts that have no
    external source (no Wikipedia page, no official docs URL).
    This guarantees every concept contributes at least its definition to the corpus.
    """
    out_path = corpus_dir / f"{_safe_filename(concept_name)}_seed.txt"
    if out_path.exists():
        return out_path

    definition = concept_data.get("definition", "")
    if not definition:
        return None

    use_cases = concept_data.get("example_use_cases", [])
    tools = concept_data.get("related_tools", [])
    refs = concept_data.get("source_references", [])

    parts = [
        f"# {concept_name}",
        f"\nCategory: {concept_data.get('category', '')} | Difficulty: {concept_data.get('difficulty', '')}",
        f"\nSource: Seed knowledge base\n",
        f"\n## Definition\n\n{definition}",
    ]
    if use_cases:
        parts.append(f"\n## Example Use Cases\n\n" + "\n".join(f"- {u}" for u in use_cases))
    if tools:
        parts.append(f"\n## Related Tools\n\n" + ", ".join(tools))
    if refs:
        parts.append(f"\n## References\n\n" + "\n".join(f"- {r}" for r in refs))

    text = "\n".join(parts)
    out_path.write_text(text, encoding="utf-8")
    logger.info("Seed fallback: wrote %d chars for '%s'", len(text), concept_name)
    return out_path


def fetch_corpus(
    concept_names: list[str],
    corpus_dir: Path,
    sources: list[str] = ("wikipedia",),
    delay_seconds: float = 0.5,
    seed_concepts: Optional[dict] = None,
) -> dict:
    """
    Fetch corpus documents for all concept names from multiple sources:
      1. Wikipedia  — general AI concepts
      2. arXiv      — research-heavy concepts
      3. Official docs — tool-specific concepts (Claude Code, MCP, Cursor, etc.)
      4. Seed fallback — writes definition text for any concept with no external source

    Saves text files to corpus_dir. Skips files that already exist.

    Args:
        concept_names: list of AI concept names to fetch
        corpus_dir: directory to save .txt files
        sources: which sources to fetch from
        delay_seconds: polite delay between API calls
        seed_concepts: dict mapping concept name → concept dict (for seed fallback)

    Returns:
        stats dict: {fetched, skipped, failed, files}
    """
    corpus_dir.mkdir(parents=True, exist_ok=True)

    fetched, skipped, failed = 0, 0, 0
    files: list[str] = []
    done = 0

    # Track which concepts got at least one file (for seed fallback)
    covered: set[str] = set()

    # Pre-mark concepts whose corpus files already exist
    for concept_name in concept_names:
        safe = _safe_filename(concept_name)
        existing = list(corpus_dir.glob(f"{safe}_*.txt"))
        if existing:
            covered.add(concept_name)

    total_sources = len(sources) + 1  # +1 for docs pass
    total = len(concept_names) * total_sources
    arxiv_blocked = False

    for concept_name in concept_names:
        got_file = concept_name in covered

        if "wikipedia" in sources:
            done += 1
            print(f"  [{done}/{total}] Wikipedia: {concept_name}", end="\r")
            path = _fetch_wikipedia(concept_name, corpus_dir)
            if path and path.stat().st_size > 0:
                fetched += 1
                files.append(str(path))
                got_file = True
            elif path is None and WIKIPEDIA_TITLE_OVERRIDES.get(concept_name) is None and concept_name in WIKIPEDIA_TITLE_OVERRIDES:
                pass  # explicitly skipped — not a failure
            elif path is None:
                failed += 1
            else:
                skipped += 1
            time.sleep(delay_seconds)

        if "arxiv" in sources:
            done += 1
            if arxiv_blocked:
                skipped += 1
            else:
                print(f"  [{done}/{total}] arXiv:     {concept_name}", end="\r")
                try:
                    path = _fetch_arxiv(concept_name, corpus_dir)
                    if path and path.stat().st_size > 0:
                        fetched += 1
                        files.append(str(path))
                        got_file = True
                    elif path is None:
                        failed += 1
                    else:
                        skipped += 1
                except _ArxivRateLimited:
                    print()
                    print(
                        f"  ⚠ arXiv rate-limited. Skipping remaining arXiv fetches.\n"
                        f"    Re-run `ingest` tomorrow to pick up arXiv."
                    )
                    arxiv_blocked = True
                    failed += 1
                time.sleep(delay_seconds)

        # Official docs — try for any concept in the URL map
        done += 1
        if concept_name in OFFICIAL_DOCS_URLS:
            print(f"  [{done}/{total}] Docs:      {concept_name}", end="\r")
            path = _fetch_official_docs(concept_name, corpus_dir)
            if path and path.stat().st_size > 0:
                fetched += 1
                files.append(str(path))
                got_file = True
            time.sleep(delay_seconds)

        if got_file:
            covered.add(concept_name)

    print()  # clear the \r line

    # Seed fallback — write definition-based corpus for any concept still uncovered
    if seed_concepts:
        fallback_count = 0
        for concept_name in concept_names:
            if concept_name not in covered and concept_name in seed_concepts:
                path = _write_seed_fallback(concept_name, seed_concepts[concept_name], corpus_dir)
                if path:
                    fetched += 1
                    files.append(str(path))
                    fallback_count += 1
        if fallback_count:
            print(f"  ✓ Seed fallback: wrote {fallback_count} definition files for uncovered concepts")

    return {"fetched": fetched, "skipped": skipped, "failed": failed, "files": files}
