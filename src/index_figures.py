"""
Index Figures for Architecture Search Engine
Indexes figure descriptions into ChromaDB for multimodal search.

WHAT THIS DOES:
1. Load figure descriptions from figure_extractor
2. Combine title + description for better search
3. Add them to a separate collection in ChromaDB
4. Enable searching for architectures by description
"""

import json
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path


# =============================================================
# CONFIGURATION
# =============================================================

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "cv_figures"
DESCRIPTIONS_FILE = Path("data/figure_descriptions.json")
USE_OPENAI = False  # Must match embeddings.py


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
# VALIDATION
# =============================================================

def is_valid_description(desc: str) -> bool:
    """Check if description is valid (not a refusal from GPT)."""
    if not desc or len(desc) < 50:
        return False
    
    bad_phrases = [
        "unable to analyze",
        "can't analyze",
        "cannot analyze",
        "i'm unable",
        "i cannot",
        "provide a description",
        "provide more context",
        "describe it or provide",
        "i can't directly",
        "cannot directly",
        "if you describe it",
        "i'm not able",
        "cannot view",
        "can't view",
    ]
    
    desc_lower = desc.lower()
    return not any(phrase in desc_lower for phrase in bad_phrases)


# =============================================================
# INDEX FIGURES
# =============================================================

def index_figures(figures: list[dict]):
    """
    Index figure descriptions in ChromaDB.
    Combines title + description for better search.
    """
    print("="*60)
    print(" INDEXING FIGURES INTO CHROMADB")
    print("="*60)
    
    # Filter valid figures
    valid_figures = [
        f for f in figures 
        if f.get('description') and is_valid_description(f['description'])
    ]
    
    print(f"   Total figures: {len(figures)}")
    print(f"   Valid descriptions: {len(valid_figures)}")
    
    if not valid_figures:
        print("❌ No valid figures to index!")
        return None
    
    # Setup ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_fn = get_embedding_function()
    
    # Delete old collection if exists
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"   Deleted old collection: {COLLECTION_NAME}")
    except:
        pass
    
    # Create new collection
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"description": "CV paper architecture figures"}
    )
    
    # Prepare data
    ids = []
    documents = []
    metadatas = []
    
    for i, fig in enumerate(valid_figures):
        # Combine title + description for better search
        title = fig.get('title', '').strip()
        description = fig.get('description', '').strip()
        
        # Create combined search text
        if title and title != "(no title)":
            search_text = f"{title}\n\n{description}"
        else:
            search_text = description
        
        ids.append(f"fig_{i}")
        documents.append(search_text)
        metadatas.append({
            "filename": fig.get('filename', ''),
            "filepath": fig.get('filepath', ''),
            "source_pdf": fig.get('source_pdf', ''),
            "page": fig.get('page', 0),
            "width": fig.get('width', 0),
            "height": fig.get('height', 0),
            "aspect_ratio": fig.get('aspect_ratio', 0),
            "title": title[:200] if title else "",
            "description": description[:500],
        })
    
    # Add to collection
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    
    print(f" Indexed {len(valid_figures)} figures!")
    print(f"   Collection: {COLLECTION_NAME}")
    
    return collection


# =============================================================
# SEARCH FIGURES
# =============================================================

def search_figures(query: str, n_results: int = 5):
    """
    Search for architecture figures by description.
    
    Args:
        query: Natural language query (e.g., "transformer encoder decoder")
        n_results: Number of results to return
    
    Returns:
        Search results with figures and metadata
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_fn = get_embedding_function()
    
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    return results


def format_search_results(results: dict) -> list[dict]:
    """
    Format search results for display.
    """
    formatted = []
    
    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        distance = results['distances'][0][i]
        similarity = 1 - distance  # Convert distance to similarity
        
        formatted.append({
            "rank": i + 1,
            "similarity": round(similarity, 3),
            "filename": meta['filename'],
            "filepath": meta['filepath'],
            "source_pdf": meta['source_pdf'],
            "page": meta['page'],
            "title": meta.get('title', ''),
            "description": meta.get('description', ''),
        })
    
    return formatted


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":
    print("="*60)
    print(" ARCHITECTURE FIGURE INDEXER")
    print("="*60)
    
    # Load descriptions
    if not DESCRIPTIONS_FILE.exists():
        print(f"❌ File not found: {DESCRIPTIONS_FILE}")
        print("   Run figure_extractor.py first!")
        exit()
    
    print(f" Loading from {DESCRIPTIONS_FILE}...")
    with open(DESCRIPTIONS_FILE, 'r') as f:
        figures = json.load(f)
    
    print(f" Loaded {len(figures)} figures")
    
    # Index
    collection = index_figures(figures)
    
    if not collection:
        exit()
    
    # Test searches
    print("\n" + "="*60)
    print("TESTING SEARCH")
    print("="*60)
    
    test_queries = [
        "transformer encoder decoder architecture",
        "CNN backbone feature extraction",
        "multi-modal fusion network",
        "attention mechanism",
        "object detection pipeline",
    ]
    
    for query in test_queries:
        print(f" Query: \"{query}\"")
        print("-"*50)
        
        results = search_figures(query, n_results=3)
        formatted = format_search_results(results)
        
        for r in formatted:
            print(f"\n   [{r['rank']}] Similarity: {r['similarity']:.3f}")
            print(f"       File: {r['filename']}")
            print(f"       Paper: {r['source_pdf'][:40]}...")
            print(f"       Title: {r['title'][:50]}..." if r['title'] else "       Title: (none)")
            print(f"       Desc: {r['description'][:80]}...")
    
    # Summary
    print("\n" + "="*60)
    print(" INDEXING COMPLETE")
    print("="*60)
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   Figures indexed: {collection.count()}")
    print(" Ready for Architecture Search Engine! ")