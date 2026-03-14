# Graph RAG — AI Concepts Knowledge Graph

A **Graph Retrieval-Augmented Generation** system for AI terminology. Built for two audiences:
- **Learners** overwhelmed by the AI landscape — look up any term, understand it clearly, see what it relates to, and get a personalized learning path
- **Portfolio readers** — a demonstration of advanced graph-based retrieval, knowledge graph modeling, BFS traversal, PageRank, hybrid retrieval, and LLM reasoning over structured graph context

---

## What It Does

| Feature | Description |
|---|---|
| **Concept Lookup** | Type any AI term → definition, category, difficulty, use cases |
| **Relationship Exploration** | See how concepts connect: REQUIRES, IS_A, BUILT_WITH, SIMILAR_TO, etc. |
| **Interactive Graph** | Click-to-explore pyvis knowledge graph, color-coded by category, sized by PageRank |
| **Learning Path** | "How do I get from Python to GraphRAG?" — BFS over prerequisite edges |
| **Curriculum Generator** | "I know ML basics, teach me RAG" → topologically sorted curriculum |
| **Concept Comparison** | RAG vs Fine-tuning — shared prerequisites, unique dependencies |
| **Natural Language Q&A** | Ask anything → Graph RAG retrieval → local LLM answer (Ollama) |
| **Text Ingestion** | Paste any AI article → LLM extracts concepts and relationships |

---

## Architecture

```
User Query (CLI / Streamlit / FastAPI)
    │
    ▼
EntityExtractor          Keyword match (O(n)) → Ollama LLM fallback
    │
    ▼
GraphRetriever           BFS 2-hop traversal over ConceptNodes
    │         \
    │          ──▶ VectorRetriever (ChromaDB, optional)
    │
    ▼
ContextAssembler         Structured markdown: concept card + relationships + hop-2 neighbors
    │
    ▼
AnswerGenerator          Ollama (llama3) — grounded in graph context
    │
    ▼
Response

Advanced graph techniques:
  PageRank            → identify foundational concepts, size nodes in viz
  BFS learning path   → shortest prerequisite chain between two concepts
  Curriculum gen      → topological sort of prerequisite subgraph
  Community           → concept clustering (python-louvain)
```

---

## Project Structure

```
graph-rag/
├── main.py                          # CLI entry point
├── requirements.txt
├── .env.example
│
├── config/
│   └── settings.py                  # All env-driven config
│
├── src/
│   ├── graph/
│   │   ├── schema.py                # ConceptNode, Relationship, GraphSubgraph, GraphClient Protocol
│   │   ├── networkx_client.py       # Default backend — no external DB needed
│   │   ├── neo4j_client.py          # Optional Neo4j backend
│   │   ├── algorithms.py            # PageRank, BFS learning path, curriculum generation
│   │   └── queries.py               # Named traversal helpers (dependencies, comparison)
│   │
│   ├── ingestion/
│   │   ├── seed_data.py             # Load seed_concepts.json → graph + ChromaDB
│   │   └── relationship_extractor.py # Ollama → (concept, REL, concept) triples
│   │
│   ├── retrieval/
│   │   ├── entity_extractor.py      # Two-stage: keyword match → LLM fallback
│   │   ├── graph_retriever.py       # BFS traversal with mode detection
│   │   ├── vector_retriever.py      # ChromaDB semantic search (optional)
│   │   └── context_assembler.py     # Graph subgraph → structured LLM prompt
│   │
│   ├── llm/
│   │   └── answer_generator.py      # Ollama client, prompt templates, fallbacks
│   │
│   └── api/
│       ├── app.py                   # FastAPI with 10 endpoints
│       └── models.py                # Pydantic request/response schemas
│
├── dashboard/
│   ├── app.py                       # Streamlit — 4 tabs
│   └── graph_viz.py                 # pyvis → interactive HTML graph
│
└── data/
    └── seed_concepts.json           # 50 AI concepts + 80 relationships
```

---

## Graph Model

### Node: Concept
```python
ConceptNode(
    name: str,
    definition: str,          # beginner-friendly, 2-3 sentences
    category: str,            # Architecture | Technique | Model | Tool | ...
    difficulty: str,          # beginner | intermediate | advanced | expert
    example_use_cases: list[str],
    related_tools: list[str],
    source_references: list[str],
)
```

### Relationship Types
| Type | Meaning | Example |
|------|---------|---------|
| `IS_A` | Specialization | GPT IS_A Transformer |
| `PART_OF` | Component | Attention Mechanism PART_OF Transformer |
| `VARIANT_OF` | Variant | GraphRAG VARIANT_OF RAG |
| `REQUIRES` | Prerequisite | RAG REQUIRES Vector Database |
| `EXTENDS` | Enhancement | ReAct EXTENDS Chain-of-Thought |
| `BUILT_WITH` | Uses as component | LangChain BUILT_WITH LLM |
| `SIMILAR_TO` | Conceptually close | FAISS SIMILAR_TO ChromaDB |
| `ALTERNATIVE_TO` | Can replace | LoRA ALTERNATIVE_TO Fine-tuning |
| `USED_IN` | Application | Embedding USED_IN Semantic Search |

---

## Setup

### 1. Clone and install
```bash
cd graph-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env if you want Neo4j or to change the Ollama model
```

### 3. Set up Ollama (for LLM features)
```bash
# macOS
brew install ollama
ollama pull llama3        # ~4GB download
ollama serve              # runs at http://localhost:11434
```
> **Without Ollama**: The system still works — graph traversal and context assembly run fully locally. You just won't get natural language answers.

### 4. Seed the knowledge graph
```bash
python main.py seed
```

---

## Usage

### CLI
```bash
# Show graph stats and top PageRank concepts
python main.py status

# Natural language Q&A
python main.py query "What is RAG?"
python main.py query "What does GraphRAG depend on?"
python main.py query "What should I know before learning RAG?"

# Show the assembled graph context
python main.py query "What is RAG?" --show-context

# Learning path between two concepts
python main.py path "Tokenization" "GraphRAG"
python main.py path "Embedding" "ReAct" --verbose

# Generate a learning curriculum
python main.py curriculum --target GraphRAG --known "Python,Machine Learning"
python main.py curriculum --target RLHF --verbose
```

### Streamlit Dashboard
```bash
python main.py dashboard
# Open: http://localhost:8501
```

### FastAPI
```bash
python main.py serve
# Docs: http://localhost:8000/docs
```

Key API endpoints:
```
POST /query                       Full pipeline
GET  /concept/{name}              Concept details + PageRank
GET  /concept/{name}/related      2-hop neighbors
GET  /concept/{name}/dependencies Prerequisite chain
GET  /path/{from}/{to}            BFS learning path
GET  /curriculum?target=RAG&known=Python,ML
GET  /compare/{a}/{b}             Side-by-side comparison
GET  /concepts                    All concepts ranked by PageRank
POST /ingest/text                 Ingest free text → extract concepts
GET  /health                      System status
```

---

## Seed Data

50 AI concepts across 8 categories with 80+ relationships:

**Foundational:** Transformer, Attention Mechanism, LLM, Tokenization, Embedding, BERT, GPT, Token, Context Window

**Retrieval & RAG:** RAG, GraphRAG, Semantic Search, Dense Retrieval, Sparse Retrieval, BM25, Hybrid Search, HyDE, Multi-hop Reasoning, Reranker, Chunking, Document Loader

**Vector & Storage:** Vector Database, FAISS, ChromaDB, Pinecone, Embedding Model, SentenceTransformers

**Training & Adaptation:** Fine-tuning, LoRA, QLoRA, PEFT, RLHF, Instruction Tuning, Few-shot Learning, Zero-shot Learning, In-context Learning

**Agents & Reasoning:** Agent, Tool Use, ReAct, Chain-of-Thought, Prompt Engineering, Hallucination, Grounding

**Frameworks:** LangChain, LlamaIndex, Ollama, Neo4j, Knowledge Graph

**Evaluation:** Faithfulness, RAGAS, Cross-encoder, Bi-encoder, KV Cache

---

## Advanced Features Showcased

| Technique | Where |
|---|---|
| **Graph traversal (BFS)** | `networkx_client.py:get_neighbors()` |
| **PageRank centrality** | `algorithms.py:get_concept_importance()` |
| **Topological sort** | `algorithms.py:generate_curriculum()` |
| **Multi-hop reasoning** | Graph traversal depth 2-4 hops |
| **Subgraph-conditioned prompting** | `context_assembler.py:assemble_context()` |
| **Hybrid retrieval** | Graph primary + ChromaDB supplementary |
| **Query intent detection** | `graph_retriever.py:detect_mode()` |
| **Two-stage entity extraction** | `entity_extractor.py` keyword + LLM |
| **Backend abstraction (Protocol)** | `schema.py:GraphClient` + factory |
| **Relationship extraction** | `relationship_extractor.py` (Ollama) |

---

## Neo4j Backend (Optional)

```bash
# Start Neo4j (Docker)
docker run -p 7687:7687 -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5

# Configure .env
GRAPH_BACKEND=neo4j
NEO4J_URI=bolt://localhost:7687

# Re-seed
python main.py seed
```

---

## Extending the Graph

Ingest any AI article or blog post:
```bash
# Via CLI (future: file flag)
# Via API:
curl -X POST http://localhost:8000/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Mixture of Experts (MoE) is a technique where..."}'
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Graph database | NetworkX (default) / Neo4j |
| Graph algorithms | NetworkX (PageRank, BFS, topological sort) |
| Vector store | ChromaDB |
| Embeddings | SentenceTransformers `all-MiniLM-L6-v2` |
| LLM | Ollama (llama3, local, free) |
| API | FastAPI |
| Dashboard | Streamlit + pyvis |
| Config | python-dotenv |
