"""
Embeddings & Vector Store
Transforms text chunks into vectors and stores them in ChromaDB.

WHAT THIS DOES:
1. Takes chunks (text) from document_processor
2. Converts each chunk to a vector (embedding)
3. Stores vectors in ChromaDB for fast similarity search
"""

import chromadb
from chromadb.utils import embedding_functions
from document_processor import process_papers_folder
from chromadb.errors import NotFoundError

# =============================================================
# CONFIGURATION
# =============================================================

# Option 1: OpenAI embeddings (best quality, paid)
# Requires: uv pip install openai
# OPENAI_API_KEY must be in .env

# Option 2: Local embeddings (free, slightly worse but sufficient)
# Uses sentence-transformers, runs on your CPU

USE_OPENAI = False  # Set to True if you want to use OpenAI


# =============================================================
# SETUP EMBEDDING FUNCTION
# =============================================================

def get_embedding_function():
    """
    Returns the embedding function to use.
    
    OpenAI: text-embedding-ada-002 (1536 dimensions)
    Local: all-MiniLM-L6-v2 (384 dimensions, but free!)
    """
    if USE_OPENAI:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-ada-002"
        )
    else:
        # Free local model
        # First call = downloads the model (~90MB)
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

# =============================================================
# CREATE VECTOR STORE
# =============================================================

def create_vector_store(chunks: list, collection_name: str = "cv_papers"):
    """
    Creates a vector database from chunks.
    
    Args:
        chunks: List of Documents (from document_processor)
        collection_name: Name of the collection in ChromaDB
    
    Returns:
        collection: The ChromaDB collection ready to be queried
    """
    print(" Setting up ChromaDB...")
    
    # 1. Create ChromaDB client (local storage in ./chroma_db)
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # 2. Delete collection if it exists (to start fresh)
    try:
        client.delete_collection(collection_name)
        print(f"   Deleted existing collection '{collection_name}'")
    except NotFoundError:
        pass
    
    # 3. Create collection with our embedding function
    embedding_fn = get_embedding_function()
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"description": "Computer Vision research papers"}
    )
    
    print(f"   Collection '{collection_name}' created")
    print(f"   Embedding model: {'OpenAI' if USE_OPENAI else 'all-MiniLM-L6-v2 (local)'}")
    
    # 4. Prepare data for ChromaDB
    print(f"\n📊 Preparing {len(chunks)} chunks...")
    
    documents = []
    metadatas = []
    ids = []
    skipped = 0
    chunk_index = 0
    
    for chunk in chunks:
        content = chunk.page_content
        
        # Skip invalid chunks
        if content is None or not isinstance(content, str):
            skipped += 1
            continue
        
        # Aggressive cleaning: keep only printable ASCII and common unicode
        # Remove all control characters and problematic bytes
        cleaned = ""
        for char in content:
            # Keep only printable characters (ASCII 32-126) and common unicode
            if 32 <= ord(char) <= 126 or char in '\n\t' or ord(char) > 127:
                # Additional check: skip surrogate characters
                if not (0xD800 <= ord(char) <= 0xDFFF):
                    cleaned += char
        
        content = cleaned.strip()
        
        # Skip empty after cleaning
        if len(content) < 10:  # Minimum 10 chars
            skipped += 1
            continue
        
        documents.append(content)
        
        # Clean metadata
        clean_metadata = {
            "source_file": str(chunk.metadata.get("source_file", "unknown")),
            "page": int(chunk.metadata.get("page", 0)),
            "title": str(chunk.metadata.get("title", "unknown"))[:100],
        }
        metadatas.append(clean_metadata)
        
        ids.append(f"chunk_{chunk_index}")
        chunk_index += 1
    
    if skipped > 0:
        print(f"   Skipped {skipped} invalid chunks")
    
    print(f"    Valid chunks: {len(documents)}")
    
    # 5. Add to collection with error handling per batch
    print(" Computing embeddings and storing in ChromaDB...")
    print("   This may take a few minutes...")
    
    batch_size = 50  # Smaller batches to isolate errors
    failed_batches = 0
    
    for i in range(0, len(documents), batch_size):
        end_idx = min(i + batch_size, len(documents))
        
        try:
            collection.add(
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
            print(f"   Added chunks {i+1} to {end_idx}")
            
        except Exception as e:
            failed_batches += 1
            print(f"   Failed batch {i+1}-{end_idx}: {str(e)[:50]}")
            
            # Try adding one by one to save what we can
            for j in range(i, end_idx):
                try:
                    collection.add(
                        documents=[documents[j]],
                        metadatas=[metadatas[j]],
                        ids=[ids[j]]
                    )
                except Exception:
                    print(f"      Skipped chunk {j}")
    
    print(" Vector store created!")
    print(f"   Total vectors: {collection.count()}")
    if failed_batches > 0:
        print(f"   {failed_batches} batches had issues (partially recovered)")
    
    return collection


# =============================================================
# SEARCH FUNCTION
# =============================================================

def search_similar(collection, query: str, n_results: int = 5):
    """
    Search for chunks most similar to a query.
    
    Args:
        collection: ChromaDB collection
        query: The user's question
        n_results: Number of results to return
    
    Returns:
        results: Most similar chunks with their scores
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    return results


# =============================================================
# MAIN - Run this to test
# =============================================================

if __name__ == "__main__":
    # 1. Load chunks
    print("=" * 60)
    print("STEP 1: Loading and chunking PDFs")
    print("=" * 60)
    chunks = process_papers_folder()
    
    # 2. Create vector store
    print("\n" + "=" * 60)
    print("STEP 2: Creating vector store")
    print("=" * 60)
    collection = create_vector_store(chunks)
    
    # 3. Test a search
    print("\n" + "=" * 60)
    print("STEP 3: Testing search")
    print("=" * 60)
    
    test_query = "What is gait recognition and how does it work?"
    print(f" Query: {test_query}\n")
    
    results = search_similar(collection, test_query, n_results=3)
    
    print(" Top 3 results:")
    print("-" * 60)
    
    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        metadata = results['metadatas'][0][i]
        distance = results['distances'][0][i]
        
        # Smaller distance = more similar
        # Convert to similarity score (1 - distance for cosine)
        similarity = 1 - distance
        
        print(f"\n[{i+1}] Similarity: {similarity:.3f}")
        print(f"    Source: {metadata['source_file']}")
        print(f"    Page: {metadata['page']}")
        print(f"    Content preview: {doc[:200]}...")