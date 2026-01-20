"""
Multimodal RAG for Architecture Search Engine
Combines text search + figure search for comprehensive results.

USAGE:
    from multimodal_rag import search, search_interactive
    
    # Simple search
    results = search("transformer attention mechanism")
    
    # Interactive mode
    search_interactive()
"""

import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from typing import Optional


# =============================================================
# CONFIGURATION
# =============================================================

CHROMA_PATH = "./chroma_db"
TEXT_COLLECTION = "cv_papers"      # From embeddings.py
FIGURE_COLLECTION = "cv_figures"   # From index_figures.py
USE_OPENAI = False


# =============================================================
# EMBEDDING FUNCTION
# =============================================================

def get_embedding_function():
    if USE_OPENAI:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-ada-002"
        )
    else:
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )


# =============================================================
# SEARCH FUNCTIONS
# =============================================================

def search_text(query: str, n_results: int = 5) -> list[dict]:
    """
    Search in text chunks from papers.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_fn = get_embedding_function()
    
    try:
        collection = client.get_collection(
            name=TEXT_COLLECTION,
            embedding_function=embedding_fn
        )
    except:
        print(f"Text collection '{TEXT_COLLECTION}' not found")
        return []
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    formatted = []
    for i in range(len(results['documents'][0])):
        meta = results['metadatas'][0][i]
        distance = results['distances'][0][i]
        
        # Get source from metadata - try both 'source_file' and 'source'
        source = meta.get('source_file') or meta.get('source', 'unknown')
        # If source is a full path, extract just the filename
        if '/' in str(source) or '\\' in str(source):
            source = Path(source).name
        
        formatted.append({
            "type": "text",
            "content": results['documents'][0][i],
            "source": source,
            "page": meta.get('page', 0),
            "similarity": round(1 - distance, 3),
        })
    
    return formatted


def search_figures(query: str, n_results: int = 3) -> list[dict]:
    """
    Search in figure descriptions.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_fn = get_embedding_function()
    
    try:
        collection = client.get_collection(
            name=FIGURE_COLLECTION,
            embedding_function=embedding_fn
        )
    except:
        print(f"Figure collection '{FIGURE_COLLECTION}' not found")
        return []
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    formatted = []
    for i in range(len(results['documents'][0])):
        meta = results['metadatas'][0][i]
        distance = results['distances'][0][i]
        formatted.append({
            "type": "figure",
            "filename": meta.get('filename', ''),
            "filepath": meta.get('filepath', ''),
            "source": meta.get('source_pdf', 'unknown'),
            "page": meta.get('page', 0),
            "title": meta.get('title', ''),
            "description": meta.get('description', ''),
            "similarity": round(1 - distance, 3),
        })
    
    return formatted


def search(query: str, n_text: int = 5, n_figures: int = 3) -> dict:
    """
    Multimodal search: combines text and figure results.
    
    Args:
        query: Natural language query
        n_text: Number of text results
        n_figures: Number of figure results
    
    Returns:
        Dict with 'text_results', 'figure_results', and 'combined'
    """
    text_results = search_text(query, n_text)
    figure_results = search_figures(query, n_figures)
    
    # Combine and sort by similarity
    combined = []
    
    for r in text_results:
        combined.append({
            **r,
            "display_type": "Text"
        })
    
    for r in figure_results:
        combined.append({
            **r,
            "display_type": "Figure"
        })
    
    # Sort by similarity (highest first)
    combined.sort(key=lambda x: x['similarity'], reverse=True)
    
    return {
        "query": query,
        "text_results": text_results,
        "figure_results": figure_results,
        "combined": combined,
        "stats": {
            "text_count": len(text_results),
            "figure_count": len(figure_results),
            "total": len(combined),
        }
    }


# =============================================================
# DISPLAY FUNCTIONS
# =============================================================

def display_results(results: dict, show_content: bool = True):
    """
    Pretty print search results.
    """
    print("\n" + "="*70)
    print(f"SEARCH: \"{results['query']}\"")
    print("="*70)
    
    stats = results['stats']
    print(f"   Found: {stats['text_count']} text chunks, {stats['figure_count']} figures")
    
    # Display combined results
    print("\n" + "-"*70)
    print("COMBINED RESULTS (sorted by relevance)")
    print("-"*70)
    
    for i, r in enumerate(results['combined'][:8]):  # Top 8
        print(f"\n[{i+1}] {r['display_type']} | Similarity: {r['similarity']:.3f}")
        print(f"    Source: {r['source'][:50]}...")
        print(f"    Page: {r['page']}")
        
        if r['type'] == 'text' and show_content:
            content = r['content'][:200].replace('\n', ' ')
            print(f"    Content: {content}...")
        
        elif r['type'] == 'figure':
            if r.get('title'):
                print(f"    Title: {r['title'][:60]}...")
            if r.get('description') and show_content:
                print(f"    Description: {r['description'][:100]}...")
            print(f"    File: {r['filename']}")
    
    # Display figures separately (with image paths)
    if results['figure_results']:
        print("\n" + "-"*70)
        print("ARCHITECTURE FIGURES")
        print("-"*70)
        
        for i, fig in enumerate(results['figure_results']):
            print(f"\n   [{i+1}] {fig['filename']}")
            print(f"       Paper: {fig['source'][:45]}...")
            print(f"       Path: {fig['filepath']}")
            if fig.get('title'):
                print(f"       Title: {fig['title'][:50]}...")


def search_and_display(query: str, n_text: int = 5, n_figures: int = 3):
    """
    Search and display results in one call.
    """
    results = search(query, n_text, n_figures)
    display_results(results)
    return results


# =============================================================
# INTERACTIVE MODE
# =============================================================

def search_interactive():
    """
    Interactive search mode for testing.
    """
    print("\n" + "="*70)
    print("ARCHITECTURE SEARCH ENGINE")
    print("="*70)
    print("Search for CV architectures using natural language.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    # Show example queries
    print("Example queries:")
    print("  - transformer encoder decoder")
    print("  - object detection CNN backbone")
    print("  - multi-camera tracking")
    print("  - attention mechanism for video")
    print("  - image segmentation architecture")
    print()
    
    while True:
        try:
            query = input("Search: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! ")
            break
        
        if not query:
            continue
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye! ")
            break
        
        search_and_display(query)
        print()


# =============================================================
# API-READY FUNCTIONS
# =============================================================

def search_api(query: str, n_text: int = 5, n_figures: int = 3) -> dict:
    """
    API-ready search function.
    Returns JSON-serializable results.
    """
    results = search(query, n_text, n_figures)
    
    # Clean up for JSON serialization
    api_response = {
        "query": results["query"],
        "stats": results["stats"],
        "results": []
    }
    
    for r in results["combined"]:
        item = {
            "type": r["type"],
            "similarity": r["similarity"],
            "source": r["source"],
            "page": r["page"],
        }
        
        if r["type"] == "text":
            item["content"] = r["content"]
        else:
            item["filename"] = r["filename"]
            item["filepath"] = r["filepath"]
            item["title"] = r.get("title", "")
            item["description"] = r.get("description", "")
        
        api_response["results"].append(item)
    
    return api_response


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":
    import sys
    
    # Check collections exist
    print("Checking collections...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collections = [c.name for c in client.list_collections()]
    
    if TEXT_COLLECTION not in collections:
        print(f" Text collection '{TEXT_COLLECTION}' not found!")
        print("   Run: python src/embeddings.py")
        sys.exit(1)
    
    if FIGURE_COLLECTION not in collections:
        print(f" Figure collection '{FIGURE_COLLECTION}' not found!")
        print("   Run: python src/index_figures.py")
        sys.exit(1)
    
    text_col = client.get_collection(TEXT_COLLECTION)
    fig_col = client.get_collection(FIGURE_COLLECTION)
    
    print(f"Text collection: {text_col.count()} chunks")
    print(f"Figure collection: {fig_col.count()} figures")
    
    # Run interactive mode or single query
    if len(sys.argv) > 1:
        # Single query from command line
        query = " ".join(sys.argv[1:])
        search_and_display(query)
    else:
        # Interactive mode
        search_interactive()