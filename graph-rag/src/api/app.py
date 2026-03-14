"""
FastAPI application for the Graph RAG system.

Endpoints:
  POST /query                       Full pipeline → LLM answer
  GET  /concept/{name}              Concept details + PageRank
  GET  /concept/{name}/related      2-hop neighbors
  GET  /concept/{name}/dependencies Prerequisites subgraph
  GET  /path/{from_concept}/{to}    BFS learning path
  GET  /curriculum                  Ordered learning curriculum
  GET  /compare/{a}/{b}             Side-by-side concept comparison
  GET  /concepts                    All concepts with metadata
  POST /ingest/text                 Free text → extract + ingest
  GET  /health                      System health check
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from config.settings import OLLAMA_MODEL
from src.api.models import (
    ComparisonResponse,
    ConceptListResponse,
    ConceptResponse,
    CurriculumResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    LearningPathResponse,
    QueryRequest,
    QueryResponse,
    RelationshipResponse,
    SubgraphResponse,
)
from src.graph import get_graph_client
from src.graph.algorithms import (
    find_learning_path,
    generate_curriculum,
    get_concept_importance,
)
from src.graph.queries import get_comparison
from src.ingestion.relationship_extractor import ingest_text
from src.llm.answer_generator import AnswerGenerator
from src.retrieval.context_assembler import assemble_context
from src.retrieval.entity_extractor import EntityExtractor
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.vector_retriever import VectorRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Graph RAG — AI Concepts Knowledge Graph",
    description=(
        "A Graph Retrieval-Augmented Generation system for AI terminology. "
        "Look up any AI concept, explore its relationships, discover prerequisites, "
        "and generate grounded explanations via a local knowledge graph."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared singletons (initialized at startup) ───────────────────────────────
_graph_client = None
_vector_retriever = None
_answer_generator = None
_entity_extractor = None
_graph_retriever = None
_pagerank_scores: dict[str, float] = {}


@app.on_event("startup")
async def startup():
    global _graph_client, _vector_retriever, _answer_generator
    global _entity_extractor, _graph_retriever, _pagerank_scores

    _graph_client = get_graph_client()
    _vector_retriever = VectorRetriever()
    _answer_generator = AnswerGenerator()

    names = _graph_client.get_all_concept_names()
    _entity_extractor = EntityExtractor(
        names,
        ollama_client=_answer_generator.get_client(),
        model=OLLAMA_MODEL,
    )
    _graph_retriever = GraphRetriever(_graph_client)

    # Compute PageRank if NetworkX backend
    try:
        from src.graph.networkx_client import NetworkXClient
        if isinstance(_graph_client, NetworkXClient):
            _pagerank_scores = get_concept_importance(_graph_client)
    except Exception as e:
        logger.warning("PageRank computation failed: %s", e)

    logger.info("Graph RAG API ready. Graph nodes: %d", len(names))


# ── Helper ───────────────────────────────────────────────────────────────────

def _concept_to_response(concept, pagerank: float = None) -> ConceptResponse:
    return ConceptResponse(
        **concept.to_dict(),
        pagerank_score=pagerank,
    )


def _subgraph_to_response(subgraph) -> SubgraphResponse:
    return SubgraphResponse(
        center=_concept_to_response(
            subgraph.center,
            _pagerank_scores.get(subgraph.center.name),
        ),
        nodes=[
            _concept_to_response(n, _pagerank_scores.get(n.name))
            for n in subgraph.nodes
        ],
        relationships=[
            RelationshipResponse(source=r.source, rel_type=r.rel_type, target=r.target)
            for r in subgraph.relationships
        ],
        hop_depth=subgraph.hop_depth,
    )


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    stats = _graph_client.get_stats()
    return HealthResponse(
        status="ok",
        graph_backend=stats["backend"],
        graph_nodes=stats["nodes"],
        graph_edges=stats["edges"],
        vector_docs=_vector_retriever.count(),
        ollama_available=_answer_generator.available,
        ollama_model=OLLAMA_MODEL,
    )


@app.post("/query", response_model=QueryResponse)
async def query_pipeline(req: QueryRequest):
    """
    Full Graph RAG pipeline:
    query → entity extraction → graph retrieval → context assembly → LLM answer
    """
    # 1. Entity extraction
    concept_name, confidence = _entity_extractor.extract(req.query)
    if concept_name is None:
        raise HTTPException(
            status_code=404,
            detail=f"Could not identify an AI concept in: '{req.query}'. "
                   "Try being more specific, e.g. 'What is RAG?'",
        )

    # 2. Auto-detect retrieval mode
    mode = req.mode or GraphRetriever.detect_mode(req.query)

    # 3. Graph retrieval
    try:
        subgraph = _graph_retriever.retrieve(concept_name, mode=mode, hops=req.hops)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # 4. Optional vector augmentation
    vector_results = _vector_retriever.search(
        req.query,
        exclude_names=[concept_name],
    )

    # 5. Context assembly
    context = assemble_context(subgraph, vector_results=vector_results)

    # 6. LLM answer generation
    sources = [n.name for n in subgraph.nodes[:5]]
    result = _answer_generator.generate(req.query, context, concept_name, sources)

    return QueryResponse(
        query=result.query,
        concept=result.concept,
        answer=result.answer,
        sources=result.sources,
        context_used=result.context_used,
        error=result.error or None,
    )


@app.get("/concept/{name}", response_model=ConceptResponse)
async def get_concept(name: str):
    concept = _graph_client.get_concept(name)
    if concept is None:
        raise HTTPException(status_code=404, detail=f"Concept '{name}' not found.")
    return _concept_to_response(concept, _pagerank_scores.get(concept.name))


@app.get("/concept/{name}/related", response_model=SubgraphResponse)
async def get_related(name: str, hops: int = Query(2, ge=1, le=4)):
    concept = _graph_client.get_concept(name)
    if concept is None:
        raise HTTPException(status_code=404, detail=f"Concept '{name}' not found.")
    subgraph = _graph_retriever.retrieve(concept.name, mode="standard", hops=hops)
    return _subgraph_to_response(subgraph)


@app.get("/concept/{name}/dependencies", response_model=SubgraphResponse)
async def get_dependencies(name: str):
    concept = _graph_client.get_concept(name)
    if concept is None:
        raise HTTPException(status_code=404, detail=f"Concept '{name}' not found.")
    subgraph = _graph_retriever.retrieve_prerequisites(concept.name)
    return _subgraph_to_response(subgraph)


@app.get("/path/{from_concept}/{to_concept}", response_model=LearningPathResponse)
async def get_learning_path(from_concept: str, to_concept: str):
    for name in [from_concept, to_concept]:
        if _graph_client.get_concept(name) is None:
            raise HTTPException(status_code=404, detail=f"Concept '{name}' not found.")

    try:
        from src.graph.networkx_client import NetworkXClient
        if isinstance(_graph_client, NetworkXClient):
            path = find_learning_path(_graph_client, from_concept, to_concept)
        else:
            path = [from_concept, to_concept]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return LearningPathResponse(
        from_concept=from_concept,
        to_concept=to_concept,
        path=path,
        path_length=len(path),
    )


@app.get("/curriculum", response_model=CurriculumResponse)
async def get_curriculum(
    target: str = Query(..., description="Target concept to learn"),
    known: str = Query("", description="Comma-separated list of already-known concepts"),
):
    concept = _graph_client.get_concept(target)
    if concept is None:
        raise HTTPException(status_code=404, detail=f"Concept '{target}' not found.")

    known_list = [k.strip() for k in known.split(",") if k.strip()]

    try:
        from src.graph.networkx_client import NetworkXClient
        if isinstance(_graph_client, NetworkXClient):
            curriculum = generate_curriculum(_graph_client, concept.name, known_list)
        else:
            curriculum = [concept.name]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Enrich with definitions
    curriculum_rich = []
    for c_name in curriculum:
        c = _graph_client.get_concept(c_name)
        if c:
            curriculum_rich.append({
                "name": c.name,
                "definition": c.definition,
                "difficulty": c.difficulty,
                "category": c.category,
            })

    return CurriculumResponse(
        target=concept.name,
        known_concepts=known_list,
        curriculum=curriculum,
        curriculum_with_definitions=curriculum_rich,
    )


@app.get("/compare/{concept_a}/{concept_b}", response_model=ComparisonResponse)
async def compare_concepts(concept_a: str, concept_b: str):
    for name in [concept_a, concept_b]:
        if _graph_client.get_concept(name) is None:
            raise HTTPException(status_code=404, detail=f"Concept '{name}' not found.")
    try:
        comparison = get_comparison(_graph_client, concept_a, concept_b)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    node_a = _graph_client.get_concept(concept_a)
    node_b = _graph_client.get_concept(concept_b)

    return ComparisonResponse(
        concept_a=_concept_to_response(node_a, _pagerank_scores.get(node_a.name)),
        concept_b=_concept_to_response(node_b, _pagerank_scores.get(node_b.name)),
        shared_prerequisites=comparison["shared_prerequisites"],
        unique_to_a=comparison["unique_to_a"],
        unique_to_b=comparison["unique_to_b"],
    )


@app.get("/concepts", response_model=ConceptListResponse)
async def list_concepts(
    category: str = Query(None, description="Filter by category"),
    difficulty: str = Query(None, description="Filter by difficulty level"),
):
    names = _graph_client.get_all_concept_names()
    concepts = []
    for name in names:
        c = _graph_client.get_concept(name)
        if c is None:
            continue
        if category and c.category != category:
            continue
        if difficulty and c.difficulty != difficulty:
            continue
        concepts.append({
            "name": c.name,
            "category": c.category,
            "difficulty": c.difficulty,
            "pagerank_score": _pagerank_scores.get(c.name, 0.0),
        })
    # Sort by pagerank (most central first)
    concepts.sort(key=lambda x: x["pagerank_score"], reverse=True)
    return ConceptListResponse(concepts=concepts, total=len(concepts))


@app.post("/ingest/text", response_model=IngestResponse)
async def ingest_text_endpoint(req: IngestRequest):
    if not _answer_generator.available:
        raise HTTPException(
            status_code=503,
            detail="Ollama not available — required for relationship extraction.",
        )
    stats = ingest_text(
        req.text,
        _graph_client,
        _answer_generator.get_client(),
        OLLAMA_MODEL,
    )
    # Refresh entity extractor with new concept names
    _entity_extractor.refresh_names(_graph_client.get_all_concept_names())
    return IngestResponse(**stats)


@app.get("/search")
async def search_concepts(q: str = Query(..., description="Search query")):
    results = _graph_client.search_concepts(q, limit=10)
    return {
        "query": q,
        "results": [
            {**c.to_dict(), "pagerank_score": _pagerank_scores.get(c.name, 0.0)}
            for c in results
        ],
    }
