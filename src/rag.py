"""
RAG Pipeline
Connects retrieval (ChromaDB) with generation (LLM).

WHAT THIS DOES:
1. Takes a user question
2. Finds relevant chunks from the vector store
3. Builds a prompt with context + question
4. Sends to LLM and returns the answer
"""

import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

load_dotenv()


# =============================================================
# CONFIGURATION
# =============================================================

COLLECTION_NAME = "cv_papers"
TOP_K = 5  # Number of chunks to retrieve
USE_OPENAI_EMBEDDINGS = False  # Must match what you used in embeddings.py


# =============================================================
# LOAD VECTOR STORE
# =============================================================

def get_collection():
    """
    Load the existing ChromaDB collection.
    """
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Use same embedding function as when we created the store
    if USE_OPENAI_EMBEDDINGS:
        embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-ada-002"
        )
    else:
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )
    
    return collection


# =============================================================
# RETRIEVAL
# =============================================================

def retrieve_context(collection, query: str, top_k: int = TOP_K) -> str:
    """
    Retrieve the most relevant chunks for a query.
    
    Args:
        collection: ChromaDB collection
        query: User's question
        top_k: Number of chunks to retrieve
    
    Returns:
        context: Concatenated text from top chunks
    """
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    
    # Build context string with source info
    context_parts = []
    
    for i, (doc, metadata) in enumerate(zip(
        results['documents'][0], 
        results['metadatas'][0]
    )):
        source = metadata.get('source_file', 'Unknown')
        page = metadata.get('page', '?')
        
        context_parts.append(
            f"[Source: {source}, Page {page}]\n{doc}"
        )
    
    context = "\n\n---\n\n".join(context_parts)
    
    return context


# =============================================================
# PROMPT BUILDING
# =============================================================

def build_prompt(context: str, question: str) -> str:
    """
    Build the prompt for the LLM.
    
    This is where the "magic" of RAG happens:
    We give the LLM context it never saw during training.
    """
    prompt = f"""You are a helpful research assistant specialized in Computer Vision.
Answer the question based ONLY on the provided context from research papers.
If the context doesn't contain enough information to answer, say so.
Always cite which paper/page your information comes from.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
    
    return prompt


# =============================================================
# GENERATION (LLM)
# =============================================================

def generate_answer(prompt: str) -> str:
    """
    Send prompt to LLM and get answer.
    
    Using OpenAI GPT-4o-mini (cheap and good).
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,  # Lower = more focused/factual
        max_tokens=500
    )
    
    return response.choices[0].message.content


# =============================================================
# MAIN RAG FUNCTION
# =============================================================

def ask(question: str, verbose: bool = False) -> str:
    """
    Main RAG pipeline: Question → Retrieve → Generate → Answer
    
    Args:
        question: User's question
        verbose: If True, print intermediate steps
    
    Returns:
        answer: LLM-generated answer based on retrieved context
    """
    # 1. Load collection
    collection = get_collection()
    
    # 2. Retrieve relevant chunks
    if verbose:
        print(f"\n Retrieving context for: {question}")
    
    context = retrieve_context(collection, question)
    
    if verbose:
        print(f"\n Context retrieved ({len(context)} chars)")
        print("-" * 50)
        print(context[:500] + "..." if len(context) > 500 else context)
        print("-" * 50)
    
    # 3. Build prompt
    prompt = build_prompt(context, question)
    
    if verbose:
        print(f" Prompt built ({len(prompt)} chars)")
    
    # 4. Generate answer
    if verbose:
        print(" Generating answer...")
    
    answer = generate_answer(prompt)
    
    return answer


# =============================================================
# MAIN - Interactive mode
# =============================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" RAG System for Computer Vision Papers")
    print("=" * 60)
    print("Ask questions about the papers in your database.")
    print("Type 'quit' to exit.\n")
    
    while True:
        question = input("\n Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! 👋")
            break
        
        if not question:
            continue
        
        try:
            answer = ask(question, verbose=True)
            print(f"\n✅ ANSWER:\n{answer}")
        except Exception as e:
            print(f"\n❌ Error: {e}")

