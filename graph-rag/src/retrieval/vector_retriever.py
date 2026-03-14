"""
Vector retriever — optional supplementary retrieval via ChromaDB.

Used to augment graph context with semantically similar concept definitions
when the graph traversal alone may miss relevant supporting text.
"""
from __future__ import annotations

import logging
from typing import Optional

from config.settings import CHROMA_COLLECTION, CHROMA_PERSIST_DIR, EMBEDDING_MODEL, VECTOR_TOP_K

logger = logging.getLogger(__name__)


class VectorRetriever:
    """
    ChromaDB-based semantic search over concept definitions.

    Gracefully degrades: if chromadb or sentence-transformers are unavailable,
    all methods return empty results without raising exceptions.
    """

    def __init__(
        self,
        collection_name: str = CHROMA_COLLECTION,
        persist_dir: str = str(CHROMA_PERSIST_DIR),
        embedding_model: str = EMBEDDING_MODEL,
        top_k: int = VECTOR_TOP_K,
    ):
        self._top_k = top_k
        self._collection = None
        self._available = False
        self._init(collection_name, persist_dir, embedding_model)

    def _init(self, collection_name: str, persist_dir: str, embedding_model: str) -> None:
        try:
            import chromadb
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

            ef = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
            client = chromadb.PersistentClient(path=persist_dir)
            self._collection = client.get_or_create_collection(
                name=collection_name,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )
            self._available = True
            logger.info(
                "Vector store initialized: %d documents in collection '%s'.",
                self._collection.count(), collection_name,
            )
        except ImportError:
            logger.warning(
                "chromadb or sentence-transformers not installed. "
                "Vector retrieval disabled."
            )
        except Exception as e:
            logger.warning("Failed to initialize vector store: %s", e)

    @property
    def available(self) -> bool:
        return self._available

    def add_documents(
        self,
        documents: list[str],
        ids: list[str],
        metadatas: Optional[list[dict]] = None,
    ) -> None:
        """Add documents to the ChromaDB collection."""
        if not self._available:
            return
        try:
            # Upsert in batches to avoid memory issues
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                batch_meta = metadatas[i:i + batch_size] if metadatas else None
                self._collection.upsert(
                    documents=batch_docs,
                    ids=batch_ids,
                    metadatas=batch_meta,
                )
            logger.info("Indexed %d documents into vector store.", len(documents))
        except Exception as e:
            logger.warning("Failed to add documents to vector store: %s", e)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        exclude_names: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Semantic similarity search.

        Returns list of dicts: [{"text": ..., "metadata": ..., "distance": ...}]
        Returns [] if vector store is unavailable.
        """
        if not self._available:
            return []

        k = top_k or self._top_k
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(k, self._collection.count() or 1),
                include=["documents", "metadatas", "distances"],
            )
            hits = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                name = meta.get("name", "")
                if exclude_names and name in exclude_names:
                    continue
                hits.append({
                    "text": doc,
                    "metadata": meta,
                    "distance": dist,
                    "similarity": 1.0 - dist,  # cosine distance → similarity
                })
            return hits
        except Exception as e:
            logger.warning("Vector search failed: %s", e)
            return []

    def count(self) -> int:
        """Return number of documents in the collection."""
        if not self._available:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0
