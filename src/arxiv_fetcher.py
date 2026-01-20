"""
arXiv Paper Fetcher
Downloads Computer Vision papers from arXiv automatically.
"""

import arxiv
import os
from pathlib import Path


def fetch_papers(
    query: str,
    max_results: int = 10,
    download_dir: str = "data/papers"
) -> list[dict]:
    """
    Search and download papers from arXiv.
    
    Args:
        query: Search query (e.g., "person re-identification")
        max_results: Number of papers to fetch
        download_dir: Where to save PDFs
    
    Returns:
        List of paper metadata
    """
    
    # Create download directory if it doesn't exist
    Path(download_dir).mkdir(parents=True, exist_ok=True)
    
    # Create arXiv client
    client = arxiv.Client()
    
    # Build search query
    # cat:cs.CV = Computer Vision category
    search = arxiv.Search(
        query=f"cat:cs.CV AND ({query})",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate  # Most recent first
    )
    
    papers = []
    
    print(f" Searching arXiv for: {query}")
    print(f" Download folder: {download_dir}")
    print("-" * 50)
    
    for result in client.results(search):
        paper_info = {
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "abstract": result.summary,
            "published": result.published.strftime("%Y-%m-%d"),
            "arxiv_id": result.entry_id.split("/")[-1],
            "pdf_url": result.pdf_url,
        }
        
        # Download PDF
        print(f"\n {paper_info['title'][:60]}...")
        print(f"   Authors: {', '.join(paper_info['authors'][:3])}")
        print(f"   Published: {paper_info['published']}")
        
        try:
            # Create safe filename
            safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in result.title)
            safe_title = safe_title[:50]  # Limit length
            filename = f"{paper_info['arxiv_id']}_{safe_title}.pdf"
            filepath = os.path.join(download_dir, filename)
            
            # Download
            result.download_pdf(dirpath=download_dir, filename=filename)
            paper_info["local_path"] = filepath
            print(f"    Downloaded: {filename}")
            
        except Exception as e:
            print(f"    Download failed: {e}")
            paper_info["local_path"] = None
        
        papers.append(paper_info)
    
    print("\n" + "=" * 50)
    print(f" Fetched {len(papers)} papers")
    
    return papers

def fetch_multiple_topics(
    topics: list[str] = None,
    papers_per_topic: int = 3) -> list[dict]:
    """
    Fetch papers for multiple CV topics at once.
    
    Args:
        topics: List of search queries
        papers_per_topic: Number of papers per topic
    
    Returns:
        List of all paper metadata
    """
    if topics is None:
        # Default topics relevant to your expertise
        topics = [
            "person re-identification",
            "multi-camera tracking",
            "pedestrian detection",
            "gait recognition",
            "human pose estimation",
        ]
    
    all_papers = []
    
    print("=" * 60)
    print(" Fetching Multiple CV Topics from arXiv")
    print("=" * 60)
    
    for topic in topics:
        print(f"\n🔍 Topic: {topic}")
        papers = fetch_papers(query=topic, max_results=papers_per_topic)
        all_papers.extend(papers)
    
    print("\n" + "=" * 60)
    print(f" Total papers: {len(all_papers)}")
    print("=" * 60)
    
    return all_papers

# =============================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Custom query from command line
        query = " ".join(sys.argv[1:])
        papers = fetch_papers(query=query, max_results=20)
    else:
        # 10 topics × 15 papers = 150 papers
        topics = [
            "person re-identification",
            "multi-camera tracking", 
            "pedestrian detection",
            "gait recognition",
            "human pose estimation",
            "object detection transformer",
            "video object tracking",
            "action recognition",
            "3D human reconstruction",
            "visual SLAM",
        ]
        papers = fetch_multiple_topics(topics=topics, papers_per_topic=15)
    
    print(" Next: Run 'python src/embeddings.py' to index new papers")