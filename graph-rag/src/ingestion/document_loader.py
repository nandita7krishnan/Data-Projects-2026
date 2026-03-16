"""
Document loader — reads text files from corpus_dir and splits them into
fixed-size overlapping chunks (TextChunk objects).

Chunks are cached to data/chunks.jsonl so the process is resumable.

Usage:
    from src.ingestion.document_loader import load_corpus
    chunks = load_corpus(corpus_dir)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

from config.settings import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    CHUNKS_CACHE_PATH,
)
from src.graph.schema import TextChunk

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md"}


def _word_chunks(text: str, chunk_size: int, overlap: int) -> Iterator[str]:
    """
    Split text into overlapping word-based chunks.
    Uses word count as a proxy for token count (good enough for chunking).
    """
    words = text.split()
    if not words:
        return

    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        yield " ".join(words[start:end])
        if end == len(words):
            break
        start += chunk_size - overlap


def _load_file(file_path: Path) -> str:
    """Read a text file, returning empty string on error."""
    try:
        return file_path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception as e:
        logger.warning("Could not read %s: %s", file_path, e)
        return ""


def chunk_file(
    file_path: Path,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[TextChunk]:
    """Chunk a single file into TextChunk objects."""
    text = _load_file(file_path)
    if not text:
        return []

    chunks = []
    for i, chunk_text in enumerate(_word_chunks(text, chunk_size, overlap)):
        chunk_id = f"{file_path.name}::{i}"
        chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                source_file=file_path.name,
                chunk_index=i,
                text=chunk_text,
            )
        )
    return chunks


def load_corpus(
    corpus_dir: Path,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
    use_cache: bool = True,
) -> list[TextChunk]:
    """
    Load and chunk all text files in corpus_dir.

    If use_cache=True and chunks.jsonl already exists, loads from cache
    instead of re-reading files (fast startup for subsequent runs).

    Returns list of TextChunk objects.
    """
    if use_cache and CHUNKS_CACHE_PATH.exists():
        logger.info("Loading chunks from cache: %s", CHUNKS_CACHE_PATH)
        chunks = []
        with open(CHUNKS_CACHE_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(TextChunk.from_dict(json.loads(line)))
        logger.info("Loaded %d cached chunks.", len(chunks))
        return chunks

    if not corpus_dir.exists():
        logger.warning("Corpus directory does not exist: %s", corpus_dir)
        return []

    files = [
        p for p in sorted(corpus_dir.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        logger.warning("No supported files found in %s", corpus_dir)
        return []

    all_chunks: list[TextChunk] = []
    for file_path in files:
        file_chunks = chunk_file(file_path, chunk_size, overlap)
        all_chunks.extend(file_chunks)
        logger.debug("  %s → %d chunks", file_path.name, len(file_chunks))

    # Cache to disk
    CHUNKS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_CACHE_PATH, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk.to_dict()) + "\n")

    logger.info(
        "Chunked %d files → %d chunks, cached to %s",
        len(files), len(all_chunks), CHUNKS_CACHE_PATH,
    )
    return all_chunks


def get_corpus_stats(corpus_dir: Path) -> dict:
    """Return stats about the corpus directory."""
    if not corpus_dir.exists():
        return {"files": 0, "total_chars": 0}

    files = [
        p for p in corpus_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    total_chars = sum(p.stat().st_size for p in files)
    return {
        "files": len(files),
        "total_chars": total_chars,
        "file_names": [p.name for p in files],
    }
