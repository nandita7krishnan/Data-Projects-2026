"""
Graph RAG CLI — command-line interface for the AI Concepts Knowledge Graph.

Commands:
    seed        Load seed data into graph + vector store
    query       Run a natural language query through the full pipeline
    path        Find learning path between two concepts
    curriculum  Generate a learning curriculum for a target concept
    status      Show graph statistics
    serve       Start FastAPI server
    dashboard   Launch Streamlit dashboard
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("graph-rag")


def cmd_seed(args) -> None:
    """Seed the graph and vector store from data/seed_concepts.json."""
    from src.graph import get_graph_client
    from src.ingestion.seed_data import load_seed_concepts, seed_graph, seed_vector_store
    from src.retrieval.vector_retriever import VectorRetriever

    print("🌱 Seeding knowledge graph...")
    client = get_graph_client()
    stats = seed_graph(client)
    print(f"  ✓ Concepts added: {stats['concepts_added']}")
    print(f"  ✓ Relationships added: {stats['relationships_added']}")
    if stats["skipped_relationships"] > 0:
        print(f"  ⚠ Skipped relationships: {stats['skipped_relationships']}")

    print("\n📐 Indexing into vector store...")
    vector = VectorRetriever()
    if vector.available:
        concepts, _ = load_seed_concepts()
        n = seed_vector_store(concepts, vector)
        print(f"  ✓ Indexed {n} concept documents")
    else:
        print("  ⚠ Vector store not available (chromadb not installed)")

    print("\n✅ Seeding complete!")


def cmd_status(args) -> None:
    """Display graph statistics."""
    from src.graph import get_graph_client
    from src.graph.algorithms import get_concept_importance
    from src.retrieval.vector_retriever import VectorRetriever

    client = get_graph_client()
    stats = client.get_stats()

    print("📊 Graph RAG Status")
    print(f"  Backend:       {stats['backend'].upper()}")
    print(f"  Concepts:      {stats['nodes']}")
    print(f"  Relationships: {stats['edges']}")

    # Vector store
    vector = VectorRetriever()
    print(f"  Vector docs:   {vector.count() if vector.available else 'N/A (not installed)'}")

    # PageRank top concepts
    try:
        from src.graph.networkx_client import NetworkXClient
        if isinstance(client, NetworkXClient) and stats["nodes"] > 0:
            pr_scores = get_concept_importance(client)
            top5 = sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            print("\n  Top 5 central concepts (PageRank):")
            for name, score in top5:
                c = client.get_concept(name)
                print(f"    {name:35s} ({c.category if c else '?'}) — {score:.4f}")
    except Exception:
        pass


def cmd_query(args) -> None:
    """Run a query through the full Graph RAG pipeline."""
    from src.graph import get_graph_client
    from src.llm.answer_generator import AnswerGenerator
    from src.retrieval.context_assembler import assemble_context
    from src.retrieval.entity_extractor import EntityExtractor
    from src.retrieval.graph_retriever import GraphRetriever
    from src.retrieval.vector_retriever import VectorRetriever
    from config.settings import OLLAMA_MODEL

    query = args.query
    print(f"\n🔍 Query: {query}")
    print("─" * 60)

    client = get_graph_client()
    generator = AnswerGenerator()
    vector = VectorRetriever()

    names = client.get_all_concept_names()
    if not names:
        print("⚠  Graph is empty. Run: python main.py seed")
        return

    extractor = EntityExtractor(names, ollama_client=generator.get_client())
    retriever = GraphRetriever(client)

    # Entity extraction
    concept_name, confidence = extractor.extract(query)
    if concept_name is None:
        print("❌ Could not identify an AI concept in this query.")
        print("   Try: python main.py query \"What is RAG?\"")
        return

    print(f"🎯 Detected concept: {concept_name} (confidence: {confidence:.2f})")

    # Graph retrieval
    mode = GraphRetriever.detect_mode(query)
    subgraph = retriever.retrieve(concept_name, mode=mode)
    print(f"   Retrieved {len(subgraph.nodes)} nodes, {len(subgraph.relationships)} relationships ({mode} mode)")

    # Vector augmentation
    vector_results = vector.search(query, exclude_names=[concept_name])
    if vector_results:
        print(f"   Found {len(vector_results)} supporting vector results")

    # Context assembly + LLM
    context = assemble_context(subgraph, vector_results=vector_results)
    result = generator.generate(query, context, concept_name)

    print("\n💬 Answer:")
    print("─" * 60)
    print(result.answer)

    if result.error:
        print(f"\n⚠  Note: {result.error}")

    if args.show_context:
        print("\n📋 Graph Context Used:")
        print("─" * 60)
        print(context)


def cmd_path(args) -> None:
    """Find a learning path between two concepts."""
    from src.graph import get_graph_client
    from src.graph.algorithms import find_learning_path
    from src.graph.networkx_client import NetworkXClient

    client = get_graph_client()
    if not isinstance(client, NetworkXClient):
        print("⚠  Learning path only supported with NetworkX backend.")
        return

    print(f"\n🎓 Learning path: {args.from_concept} → {args.to_concept}")
    path = find_learning_path(client, args.from_concept, args.to_concept)

    if not path or len(path) < 2:
        print("  No clear path found. These concepts may not be connected.")
        return

    for i, step in enumerate(path):
        concept = client.get_concept(step)
        prefix = "  🎯" if step == args.to_concept else f"  {i+1}."
        print(f"{prefix} {step}")
        if concept and args.verbose:
            print(f"      {concept.short_description()}")


def cmd_curriculum(args) -> None:
    """Generate a learning curriculum for a target concept."""
    from src.graph import get_graph_client
    from src.graph.algorithms import generate_curriculum
    from src.graph.networkx_client import NetworkXClient

    client = get_graph_client()
    if not isinstance(client, NetworkXClient):
        print("⚠  Curriculum generation only supported with NetworkX backend.")
        return

    known = [k.strip() for k in args.known.split(",") if k.strip()] if args.known else []
    print(f"\n🎓 Curriculum to learn: {args.target}")
    if known:
        print(f"   Already know: {', '.join(known)}")
    print("─" * 60)

    curriculum = generate_curriculum(client, args.target, known)
    if not curriculum:
        print("🎉 You already know everything needed!")
        return

    for i, c_name in enumerate(curriculum):
        c = client.get_concept(c_name)
        prefix = "🎯" if c_name == args.target else f"{i+1:2d}."
        known_mark = " ✅" if c_name in known else ""
        print(f"  {prefix} {c_name}{known_mark}")
        if c and args.verbose:
            print(f"       [{c.category} | {c.difficulty}] {c.short_description()}")


def cmd_serve(args) -> None:
    """Start the FastAPI server."""
    import uvicorn
    from config.settings import API_HOST, API_PORT

    print(f"🚀 Starting Graph RAG API at http://{API_HOST}:{API_PORT}")
    print(f"   Docs: http://localhost:{API_PORT}/docs")
    uvicorn.run(
        "src.api.app:app",
        host=API_HOST,
        port=API_PORT,
        reload=args.reload,
    )


def cmd_dashboard(args) -> None:
    """Launch the Streamlit dashboard."""
    import subprocess

    dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
    print("🖥  Starting Streamlit dashboard...")
    print("   Open your browser at http://localhost:8501")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard_path)],
        check=True,
    )


# ── Argument parser ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Graph RAG — AI Concepts Knowledge Graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py seed
  python main.py status
  python main.py query "What is RAG?"
  python main.py query "What does GraphRAG depend on?"
  python main.py path "Tokenization" "GraphRAG"
  python main.py curriculum --target GraphRAG --known "Python,Machine Learning"
  python main.py serve
  python main.py dashboard
        """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # seed
    subparsers.add_parser("seed", help="Load seed concepts into graph + vector store")

    # status
    subparsers.add_parser("status", help="Show graph statistics")

    # query
    p_query = subparsers.add_parser("query", help="Run a natural language query")
    p_query.add_argument("query", help="Your question about an AI concept")
    p_query.add_argument("--show-context", action="store_true", help="Print the assembled graph context")

    # path
    p_path = subparsers.add_parser("path", help="Find learning path between two concepts")
    p_path.add_argument("from_concept", help="Starting concept")
    p_path.add_argument("to_concept", help="Target concept")
    p_path.add_argument("-v", "--verbose", action="store_true", help="Show definitions")

    # curriculum
    p_curr = subparsers.add_parser("curriculum", help="Generate learning curriculum")
    p_curr.add_argument("--target", required=True, help="Target concept to learn")
    p_curr.add_argument("--known", default="", help="Comma-separated known concepts")
    p_curr.add_argument("-v", "--verbose", action="store_true", help="Show definitions")

    # serve
    p_serve = subparsers.add_parser("serve", help="Start FastAPI server")
    p_serve.add_argument("--reload", action="store_true", help="Enable auto-reload")

    # dashboard
    subparsers.add_parser("dashboard", help="Launch Streamlit dashboard")

    args = parser.parse_args()

    commands = {
        "seed": cmd_seed,
        "status": cmd_status,
        "query": cmd_query,
        "path": cmd_path,
        "curriculum": cmd_curriculum,
        "serve": cmd_serve,
        "dashboard": cmd_dashboard,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
