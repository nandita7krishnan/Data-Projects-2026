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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

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
    "EXTENDS", "VARIANT_OF", "SIMILAR_TO", "USED_IN", "ALTERNATIVE_TO",
]

# --- Seed data ---
SEED_DATA_PATH = DATA_DIR / "seed_concepts.json"

# --- API ---
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# --- Dashboard colors by category ---
CATEGORY_COLORS = {
    "Architecture": "#4A90D9",
    "Technique": "#F5A623",
    "Model": "#7ED321",
    "Tool": "#9B59B6",
    "Library": "#E74C3C",
    "Framework": "#1ABC9C",
    "Concept": "#F39C12",
    "Algorithm": "#2ECC71",
    "Dataset": "#3498DB",
    "Evaluation": "#E67E22",
    "Domain": "#95A5A6",
    "Infrastructure": "#34495E",
}

DIFFICULTY_COLORS = {
    "beginner": "#2ECC71",
    "intermediate": "#F39C12",
    "advanced": "#E74C3C",
    "expert": "#8E44AD",
}
