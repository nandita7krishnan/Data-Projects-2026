"""Pydantic models for FastAPI request/response schemas."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ── Request models ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language question about an AI concept")
    mode: Optional[str] = Field(
        None,
        description="Retrieval mode: standard | prerequisite | similarity | usage"
    )
    hops: Optional[int] = Field(None, ge=1, le=4, description="Graph traversal depth (1-4)")


class IngestRequest(BaseModel):
    text: str = Field(..., description="Free text to extract concepts and relationships from")


# ── Response models ──────────────────────────────────────────────────────────

class ConceptResponse(BaseModel):
    name: str
    definition: str
    category: str
    difficulty: str
    example_use_cases: list[str]
    related_tools: list[str]
    source_references: list[str]
    pagerank_score: Optional[float] = None


class RelationshipResponse(BaseModel):
    source: str
    rel_type: str
    target: str


class SubgraphResponse(BaseModel):
    center: ConceptResponse
    nodes: list[ConceptResponse]
    relationships: list[RelationshipResponse]
    hop_depth: dict[str, int]


class QueryResponse(BaseModel):
    query: str
    concept: str
    answer: str
    sources: list[str]
    context_used: str
    error: Optional[str] = None


class LearningPathResponse(BaseModel):
    from_concept: str
    to_concept: str
    path: list[str]
    path_length: int


class CurriculumResponse(BaseModel):
    target: str
    known_concepts: list[str]
    curriculum: list[str]
    curriculum_with_definitions: list[dict]


class ComparisonResponse(BaseModel):
    concept_a: ConceptResponse
    concept_b: ConceptResponse
    shared_prerequisites: list[str]
    unique_to_a: list[str]
    unique_to_b: list[str]


class IngestResponse(BaseModel):
    triples_extracted: int
    new_concepts: int
    new_relationships: int


class HealthResponse(BaseModel):
    status: str
    graph_backend: str
    graph_nodes: int
    graph_edges: int
    vector_docs: int
    ollama_available: bool
    ollama_model: str


class ConceptListResponse(BaseModel):
    concepts: list[dict]  # [{name, category, difficulty, pagerank_score}]
    total: int
