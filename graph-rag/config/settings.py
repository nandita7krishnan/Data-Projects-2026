from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# --- Graph backend ---
GRAPH_BACKEND = os.getenv("GRAPH_BACKEND", "networkx")  # "networkx" | "neo4j"
NETWORKX_PERSIST_PATH = DATA_DIR / "graph.pkl"

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# --- LLM (Ollama) ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# --- Vector store ---
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma_db")))
CHROMA_COLLECTION = "ai_concepts"

# --- Embedding ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Retrieval ---
GRAPH_HOPS = int(os.getenv("GRAPH_HOPS", "2"))
MAX_CONTEXT_NODES = int(os.getenv("MAX_CONTEXT_NODES", "25"))
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "3"))
ENTITY_CONFIDENCE_THRESHOLD = float(os.getenv("ENTITY_CONFIDENCE_THRESHOLD", "0.8"))

# --- Relationship priority (lower index = higher priority) ---
RELATIONSHIP_PRIORITY = [
    "REQUIRES", "IS_A", "PART_OF", "BUILT_WITH",
    "EXTENDS", "VARIANT_OF", "SIMILAR_TO", "COMPLEMENTARY_TO", "USED_IN", "ALTERNATIVE_TO",
]

# --- Seed data ---
SEED_DATA_PATH = DATA_DIR / "seed_concepts.json"

# --- Corpus ingestion (GraphRAG) ---
CORPUS_DIR = DATA_DIR / "corpus"
CHUNKS_CACHE_PATH = DATA_DIR / "chunks.jsonl"
PROCESSED_CHUNKS_PATH = DATA_DIR / "processed_chunks.json"
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "300"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))

# --- Community detection (GraphRAG) ---
COMMUNITIES_PATH = DATA_DIR / "communities.json"
COMMUNITY_SUMMARIES_PATH = DATA_DIR / "community_summaries.json"
MIN_COMMUNITY_SIZE = int(os.getenv("MIN_COMMUNITY_SIZE", "3"))
COMMUNITY_RESOLUTION = float(os.getenv("COMMUNITY_RESOLUTION", "1.0"))

# --- Global retrieval (GraphRAG) ---
GLOBAL_TOP_K_COMMUNITIES = int(os.getenv("GLOBAL_TOP_K_COMMUNITIES", "5"))
GLOBAL_MAX_COMMUNITY_TOKENS = int(os.getenv("GLOBAL_MAX_COMMUNITY_TOKENS", "400"))

# --- API ---
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# --- Dashboard colors by category ---
CATEGORY_COLORS = {
    "Architecture": "#E8A0BF",   # dusty rose
    "Technique": "#FF6B6B",      # coral red
    "Model": "#6DBF8A",          # sage green
    "Tool": "#FF9999",           # light salmon
    "Library": "#C0392B",        # deep red
    "Framework": "#55A868",      # muted green
    "Concept": "#F48FB1",        # medium pink
    "Algorithm": "#4CAF50",      # medium green
    "Dataset": "#FF8080",        # light coral
    "Evaluation": "#D63031",     # vivid red
    "Domain": "#FFB3BA",         # baby pink
    "Infrastructure": "#8E1A1A", # dark red
}

DIFFICULTY_COLORS = {
    "beginner": "#4CAF50",   # medium green
    "intermediate": "#FF9999",  # light salmon pink
    "advanced": "#E74C3C",   # red
    "expert": "#C0392B",     # dark red
}
