"""
FastAPI REST API for Multimodal RAG Search Engine
With LLM-powered answer synthesis.

USAGE:
    uvicorn src.api:app --reload --port 8000
    
ENDPOINTS:
    GET  /              - Frontend
    GET  /health        - Health check
    GET  /stats         - Collection statistics
    GET  /search        - Search (returns raw results)
    POST /ask           - Ask a question (returns LLM-synthesized answer)
    GET  /figures/{name} - Serve figure images
"""

import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# =============================================================
# CONFIGURATION
# =============================================================

CHROMA_PATH = "./chroma_db"
TEXT_COLLECTION = "cv_papers"
FIGURE_COLLECTION = "cv_figures"
FIGURES_PATH = "./data/figures"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# =============================================================
# PYDANTIC MODELS
# =============================================================

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    n_text: int = Field(default=5, ge=1, le=20)
    n_figures: int = Field(default=3, ge=0, le=10)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="Question to ask about CV papers")
    n_sources: int = Field(default=3, ge=1, le=10, description="Number of source papers to use")


class SearchStats(BaseModel):
    text_count: int
    figure_count: int
    total: int


class SearchResponse(BaseModel):
    query: str
    stats: SearchStats
    results: list


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list
    figures: list


class CollectionStats(BaseModel):
    text_chunks: int
    figures: int
    embedding_model: str
    status: str


class HealthResponse(BaseModel):
    status: str
    service: str


# =============================================================
# FASTAPI APP SETUP
# =============================================================

app = FastAPI(
    title="CV Papers RAG API",
    description="""
Multimodal RAG search engine for Computer Vision research papers.

**Features:**
- Semantic search across paper content
- Architecture figure search
- LLM-powered answer synthesis
- Local embeddings (no API key for search)

**Endpoints:**
- `/search` - Raw search results
- `/ask` - Ask a question, get a synthesized answer with sources
    """,
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if Path(FIGURES_PATH).exists():
    app.mount("/figures", StaticFiles(directory=FIGURES_PATH), name="figures")


# =============================================================
# EMBEDDING FUNCTION
# =============================================================

def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


# =============================================================
# SEARCH FUNCTIONS
# =============================================================

def search_text(query: str, n_results: int = 5) -> list[dict]:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_fn = get_embedding_function()
    
    try:
        collection = client.get_collection(
            name=TEXT_COLLECTION,
            embedding_function=embedding_fn
        )
    except Exception:
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
        
        source = meta.get('source_file') or meta.get('source', 'unknown')
        if '/' in str(source) or '\\' in str(source):
            source = Path(source).name
        
        formatted.append({
            "type": "text",
            "content": results['documents'][0][i],
            "source": source,
            "page": meta.get('page', 0),
            "section": meta.get('section', 'unknown'),
            "similarity": round(1 - distance, 3),
        })
    
    return formatted


def get_figures_for_paper(paper_name: str) -> list[dict]:
    """Get all figures associated with a paper."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_fn = get_embedding_function()
    
    try:
        collection = client.get_collection(
            name=FIGURE_COLLECTION,
            embedding_function=embedding_fn
        )
    except Exception:
        return []
    
    all_data = collection.get(include=["metadatas"])
    
    # Normalize paper name for matching
    paper_name_clean = paper_name.lower().replace('.pdf', '').replace('_', ' ').replace('-', ' ')
    
    figures = []
    for meta in all_data['metadatas']:
        source = meta.get('source_pdf', '')
        source_clean = source.lower().replace('.pdf', '').replace('_', ' ').replace('-', ' ')
        
        # Match if names are similar
        if (paper_name_clean in source_clean or 
            source_clean in paper_name_clean or
            paper_name_clean[:15] == source_clean[:15]):
            figures.append({
                "filename": meta.get('filename', ''),
                "image_url": f"/figures/{meta.get('filename', '')}",
                "page": meta.get('page', 0),
                "title": meta.get('title', ''),
                "description": meta.get('description', ''),
            })
    
    return figures


def get_all_figures() -> dict:
    """Get all figures grouped by paper."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_fn = get_embedding_function()
    
    try:
        collection = client.get_collection(
            name=FIGURE_COLLECTION,
            embedding_function=embedding_fn
        )
    except Exception:
        return {}
    
    all_data = collection.get(include=["metadatas"])
    
    figures_by_paper = {}
    for meta in all_data['metadatas']:
        source = meta.get('source_pdf', 'unknown')
        filename = meta.get('filename', '')
        
        if source not in figures_by_paper:
            figures_by_paper[source] = []
        
        figures_by_paper[source].append({
            "filename": filename,
            "image_url": f"/figures/{filename}",
            "page": meta.get('page', 0),
            "title": meta.get('title', ''),
            "description": meta.get('description', ''),
        })
    
    return figures_by_paper


def search_grouped(query: str, n_results: int = 5) -> list[dict]:
    """Search and group results by paper with figures."""
    text_results = search_text(query, n_results=n_results * 2)
    figures_by_paper = get_all_figures()
    
    papers_seen = {}
    for result in text_results:
        source = result['source']
        if source not in papers_seen:
            papers_seen[source] = {
                "paper": source,
                "similarity": result['similarity'],
                "page": result['page'],
                "content": result['content'],
                "figures": figures_by_paper.get(source, [])
            }
    
    grouped = list(papers_seen.values())
    grouped.sort(key=lambda x: x['similarity'], reverse=True)
    
    return grouped[:n_results]


# =============================================================
# LLM SYNTHESIS
# =============================================================

def synthesize_answer(question: str, contexts: list[dict]) -> str:
    """
    Use LLM to synthesize an answer from retrieved contexts.
    
    Args:
        question: User's question
        contexts: List of retrieved text chunks with metadata
    
    Returns:
        Synthesized answer string
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Error: OPENAI_API_KEY not set. Please add it to your .env file."
    
    client = OpenAI(api_key=api_key)
    
    # Build context string from retrieved passages
    context_parts = []
    for i, ctx in enumerate(contexts):
        source = ctx.get('source', 'Unknown')
        page = ctx.get('page', '?')
        content = ctx.get('content', '')[:1500]  # Limit content length
        context_parts.append(f"[Source {i+1}: {source}, Page {page}]\n{content}")
    
    context_string = "\n\n---\n\n".join(context_parts)
    
    # Create prompt
    prompt = f"""You are a computer vision research assistant. Based on the following excerpts from research papers, answer the user's question.

IMPORTANT INSTRUCTIONS:
- Synthesize information from the provided sources
- Be specific and technical
- Mention which paper(s) the information comes from
- If the sources don't contain enough information, say so
- Structure your answer clearly with key points
- Focus on methods, architectures, and technical contributions

SOURCES:
{context_string}

QUESTION: {question}

ANSWER:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful computer vision research assistant. Provide clear, technical, and well-structured answers based on the provided paper excerpts."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"


# =============================================================
# API ENDPOINTS
# =============================================================

@app.get("/", tags=["Info"], include_in_schema=False)
async def root():
    return FileResponse("static/index.html")


@app.get("/api", tags=["Info"])
async def api_info():
    return {
        "service": "CV Papers RAG API",
        "version": "2.0.0",
        "description": "Multimodal RAG with LLM synthesis",
        "endpoints": {
            "search": "/search",
            "ask": "/ask",
            "stats": "/stats",
            "health": "/health",
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    return HealthResponse(status="healthy", service="cv-papers-rag")


@app.get("/stats", response_model=CollectionStats, tags=["Info"])
async def get_stats():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    try:
        text_col = client.get_collection(TEXT_COLLECTION)
        text_count = text_col.count()
    except Exception:
        text_count = 0
    
    try:
        fig_col = client.get_collection(FIGURE_COLLECTION)
        fig_count = fig_col.count()
    except Exception:
        fig_count = 0
    
    status = "ready" if text_count > 0 else "empty"
    
    return CollectionStats(
        text_chunks=text_count,
        figures=fig_count,
        embedding_model=EMBEDDING_MODEL,
        status=status
    )


@app.get("/search", response_model=SearchResponse, tags=["Search"])
async def search_get(
    q: str = Query(..., min_length=1, max_length=500, description="Search query"),
    n_text: int = Query(default=5, ge=1, le=20),
    n_figures: int = Query(default=3, ge=0, le=10),
):
    """Raw search - returns matching passages and figures."""
    grouped_results = search_grouped(q, n_text)
    total_figures = sum(len(r['figures']) for r in grouped_results)
    
    return SearchResponse(
        query=q,
        stats=SearchStats(
            text_count=len(grouped_results),
            figure_count=total_figures,
            total=len(grouped_results),
        ),
        results=grouped_results
    )


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_post(request: SearchRequest):
    """Raw search - returns matching passages and figures."""
    return await search_get(request.query, request.n_text, request.n_figures)


@app.post("/ask", response_model=AskResponse, tags=["Ask"])
async def ask_question(request: AskRequest):
    """
    Ask a question and get a synthesized answer.
    
    This endpoint:
    1. Searches for relevant passages
    2. Sends them to an LLM
    3. Returns a synthesized answer with sources and figures
    """
    # Search for relevant passages
    text_results = search_text(request.question, n_results=request.n_sources * 2)
    
    if not text_results:
        return AskResponse(
            question=request.question,
            answer="No relevant information found in the indexed papers.",
            sources=[],
            figures=[]
        )
    
    # Get top results (deduplicated by paper)
    seen_papers = set()
    top_results = []
    for result in text_results:
        if result['source'] not in seen_papers and len(top_results) < request.n_sources:
            seen_papers.add(result['source'])
            top_results.append(result)
    
    # Synthesize answer
    answer = synthesize_answer(request.question, top_results)
    
    # Get figures from source papers
    all_figures = []
    for result in top_results:
        paper_figures = get_figures_for_paper(result['source'])
        all_figures.extend(paper_figures[:2])  # Max 2 figures per paper
    
    # Format sources
    sources = [
        {
            "paper": r['source'],
            "page": r['page'],
            "similarity": r['similarity'],
            "excerpt": r['content'][:300] + "..."
        }
        for r in top_results
    ]
    
    return AskResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        figures=all_figures[:5]  # Max 5 figures total
    )


@app.get("/ask", response_model=AskResponse, tags=["Ask"])
async def ask_get(
    q: str = Query(..., min_length=1, max_length=500, description="Question"),
    n_sources: int = Query(default=3, ge=1, le=10, description="Number of sources"),
):
    """GET version of /ask for easy testing."""
    return await ask_question(AskRequest(question=q, n_sources=n_sources))


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("CV Papers RAG API v2.0")
    print("=" * 60)
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print("LLM: GPT-4o-mini")
    print("-" * 60)
    print("Endpoints:")
    print("  /search - Raw search results")
    print("  /ask    - LLM-synthesized answers")
    print("-" * 60)
    print("API docs: http://localhost:8000/docs")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)