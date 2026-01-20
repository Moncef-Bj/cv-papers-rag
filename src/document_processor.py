"""
Document Processor with Intelligent Section Detection
Loads PDFs, detects sections, keeps only useful content.

WHAT IT DOES:
1. Loads PDFs and extracts text
2. Detects paper sections (Abstract, Method, References, etc.)
3. Keeps only useful sections (Abstract, Method, Experiments, Results)
4. Filters out References, Related Work, Acknowledgments
5. Chunks the filtered content for embedding

SECTIONS KEPT:
- Abstract
- Introduction  
- Method / Methodology / Approach / Proposed Method
- Model / Architecture / Framework
- Experiments / Experimental Results
- Results / Evaluation
- Discussion
- Conclusion

SECTIONS FILTERED:
- Related Work / Prior Work / Background
- References / Bibliography
- Acknowledgments / Acknowledgements
- Appendix / Supplementary
"""

import re
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =============================================================
# SECTION CONFIGURATION
# =============================================================

# Sections to KEEP (useful content)
KEEP_SECTIONS = [
    'abstract',
    'introduction',
    'method',
    'methodology', 
    'approach',
    'proposed method',
    'proposed approach',
    'our method',
    'our approach',
    'model',
    'architecture',
    'framework',
    'network',
    'system',
    'experiment',
    'experiments',
    'experimental setup',
    'experimental results',
    'implementation',
    'result',
    'results',
    'evaluation',
    'analysis',
    'ablation',
    'discussion',
    'conclusion',
    'conclusions',
    'summary',
]

# Sections to FILTER (not useful for search)
SKIP_SECTIONS = [
    'related work',
    'related works',
    'prior work',
    'previous work',
    'background',
    'literature review',
    'reference',
    'references',
    'bibliography',
    'acknowledgment',
    'acknowledgments',
    'acknowledgement',
    'acknowledgements',
    'appendix',
    'appendices',
    'supplementary',
    'supplemental',
    'supplementary material',
]


# =============================================================
# SECTION DETECTION
# =============================================================

def detect_section(text: str) -> str:
    """
    Detect which section a chunk belongs to.
    
    Args:
        text: The chunk text
        
    Returns:
        Section name or 'unknown'
    """
    text_lower = text.lower()
    lines = text.strip().split('\n')
    
    # Check first few lines for section headers
    for line in lines[:5]:
        line_clean = line.strip().lower()
        line_clean = re.sub(r'^[\d\.\s]+', '', line_clean)  # Remove "1." or "1.1"
        line_clean = re.sub(r'[:\.]$', '', line_clean)  # Remove trailing : or .
        line_clean = line_clean.strip()
        
        # Check against skip sections
        for section in SKIP_SECTIONS:
            if line_clean == section or line_clean.startswith(section + ' '):
                return section
        
        # Check against keep sections
        for section in KEEP_SECTIONS:
            if line_clean == section or line_clean.startswith(section + ' '):
                return section
    
    return 'unknown'


def is_reference_content(text: str) -> bool:
    """
    Detect if text content is from references section based on content patterns.
    More aggressive detection for reference-style text.
    
    Args:
        text: The chunk text
        
    Returns:
        True if this looks like reference content
    """
    if not text or len(text) < 50:
        return False
    
    text_lower = text.lower()
    
    # Pattern 1: Multiple bracketed citations at start of lines
    lines = text.strip().split('\n')
    bracket_starts = sum(1 for line in lines if re.match(r'^\s*\[\d+\]', line))
    if bracket_starts >= 2:
        return True
    
    # Pattern 2: High density of author patterns "Name, A.,"
    author_patterns = re.findall(r'[A-Z][a-z]+,\s*[A-Z]\.', text)
    if len(author_patterns) >= 4:
        return True
    
    # Pattern 3: Multiple "et al." occurrences
    et_al_count = len(re.findall(r'et al\.', text_lower))
    if et_al_count >= 3:
        return True
    
    # Pattern 4: Publication patterns
    pub_patterns = [
        r'proceedings of',
        r'in proc\.',
        r'ieee trans',
        r'acm trans',
        r'arxiv preprint',
        r'arxiv:\d+\.\d+',
        r'pp\.\s*\d+[-–]\d+',
        r'vol\.\s*\d+',
        r'no\.\s*\d+',
    ]
    pub_count = sum(len(re.findall(p, text_lower)) for p in pub_patterns)
    if pub_count >= 3:
        return True
    
    # Pattern 5: Conference acronyms density
    conf_pattern = r'\b(CVPR|ICCV|ECCV|NeurIPS|ICML|AAAI|IJCAI|ACL|EMNLP|WACV|BMVC)\b'
    conf_count = len(re.findall(conf_pattern, text))
    if conf_count >= 3:
        return True
    
    # Pattern 6: Year patterns at end of sentences (typical in references)
    year_endings = len(re.findall(r',\s*\d{4}\.', text))
    if year_endings >= 3:
        return True
    
    return False


def is_related_work_content(text: str) -> bool:
    """
    Detect if text is from Related Work section based on content.
    Related work typically discusses other papers without presenting new methods.
    
    Args:
        text: The chunk text
        
    Returns:
        True if this looks like related work content
    """
    if not text or len(text) < 100:
        return False
    
    text_lower = text.lower()
    
    # Pattern 1: Many citations inline like [1], [2, 3], [4-6]
    citation_count = len(re.findall(r'\[\d+(?:[-,\s]*\d+)*\]', text))
    words = len(text.split())
    if words > 0 and citation_count / words > 0.03:  # More than 3 citations per 100 words
        return True
    
    # Pattern 2: Phrases typical of related work
    related_phrases = [
        r'\bproposed\s+(?:a|an|the)\s+method',
        r'\bintroduced\s+(?:a|an|the)',
        r'\bpresented\s+(?:a|an|the)',
        r'previous\s+(?:work|method|approach)',
        r'existing\s+(?:work|method|approach)',
        r'prior\s+(?:work|method|approach)',
        r'early\s+(?:work|method|approach)',
        r'recent\s+(?:work|method|approach)',
        r'\[[\d,\s-]+\]\s+(?:proposed|introduced|presented|used|applied)',
    ]
    
    phrase_count = sum(len(re.findall(p, text_lower)) for p in related_phrases)
    if phrase_count >= 4:
        return True
    
    return False


def is_low_quality_chunk(text: str) -> bool:
    """
    Detect low-quality chunks that should be excluded.
    
    Args:
        text: The chunk text
        
    Returns:
        True if this is a low-quality chunk
    """
    if not text:
        return True
    
    text = text.strip()
    
    # Too short
    if len(text) < 150:
        return True
    
    # Mostly numbers (tables, data)
    alpha_chars = sum(1 for c in text if c.isalpha())
    if len(text) > 0 and alpha_chars / len(text) < 0.5:
        return True
    
    # Too many special characters (equations, code)
    special_chars = sum(1 for c in text if not c.isalnum() and c not in ' .,;:!?-\n\'\"()')
    if len(text) > 0 and special_chars / len(text) > 0.25:
        return True
    
    # Looks like a table or figure caption only
    if text.lower().startswith(('table ', 'figure ', 'fig. ', 'tab. ')):
        if len(text) < 300:
            return True
    
    return False


def should_keep_chunk(text: str, section: str) -> bool:
    """
    Decide if a chunk should be kept based on section and content.
    
    Args:
        text: The chunk text
        section: Detected section name
        
    Returns:
        True if chunk should be kept
    """
    # Always skip these sections
    if section in SKIP_SECTIONS:
        return False
    
    # Check content patterns even if section is unknown
    if is_reference_content(text):
        return False
    
    if is_related_work_content(text):
        return False
    
    if is_low_quality_chunk(text):
        return False
    
    return True


# =============================================================
# PDF LOADING AND CHUNKING
# =============================================================

def load_pdf(pdf_path: str) -> list:
    """
    Load a PDF and return its pages as documents.
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        List of document objects (one per page)
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    return pages


def chunk_documents(
    documents: list,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> list:
    """
    Split documents into smaller chunks for better retrieval.
    
    Args:
        documents: List of document objects
        chunk_size: Max characters per chunk (1000 ~ 250 tokens)
        chunk_overlap: Overlap between chunks (to keep context)
    
    Returns:
        List of chunked documents
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    return chunks


# =============================================================
# MAIN PROCESSING FUNCTION
# =============================================================

def process_papers_folder(folder_path: str = "data/papers") -> list:
    """
    Process all PDFs in a folder with intelligent section filtering.
    
    Args:
        folder_path: Path to folder containing PDFs
    
    Returns:
        List of filtered chunks (only useful sections)
    """
    folder = Path(folder_path)
    all_chunks = []
    
    pdf_files = list(folder.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")
    print("-" * 60)
    
    stats = {
        'total_chunks': 0,
        'kept_chunks': 0,
        'filtered_references': 0,
        'filtered_related_work': 0,
        'filtered_low_quality': 0,
        'filtered_by_section': 0,
    }
    
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name[:50]}...")
        
        try:
            # Load PDF
            pages = load_pdf(str(pdf_path))
            print(f"   Pages: {len(pages)}")
            
            # Chunk the pages
            chunks = chunk_documents(pages)
            stats['total_chunks'] += len(chunks)
            
            # Filter chunks
            kept_chunks = []
            
            for chunk in chunks:
                content = chunk.page_content
                
                # Detect section
                section = detect_section(content)
                
                # Check if should keep
                if section in SKIP_SECTIONS:
                    stats['filtered_by_section'] += 1
                    continue
                
                if is_reference_content(content):
                    stats['filtered_references'] += 1
                    continue
                
                if is_related_work_content(content):
                    stats['filtered_related_work'] += 1
                    continue
                
                if is_low_quality_chunk(content):
                    stats['filtered_low_quality'] += 1
                    continue
                
                # Keep this chunk
                chunk.metadata["source_file"] = pdf_path.name
                chunk.metadata["section"] = section
                kept_chunks.append(chunk)
            
            stats['kept_chunks'] += len(kept_chunks)
            all_chunks.extend(kept_chunks)
            
            filtered = len(chunks) - len(kept_chunks)
            print(f"   Chunks: {len(chunks)} -> {len(kept_chunks)} (filtered {filtered})")
            
        except Exception as e:
            print(f"   Error: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total chunks processed:    {stats['total_chunks']}")
    print(f"Chunks kept:               {stats['kept_chunks']}")
    print(f"Filtered (references):     {stats['filtered_references']}")
    print(f"Filtered (related work):   {stats['filtered_related_work']}")
    print(f"Filtered (low quality):    {stats['filtered_low_quality']}")
    print(f"Filtered (by section):     {stats['filtered_by_section']}")
    print(f"Filter rate:               {100 * (1 - stats['kept_chunks']/max(1, stats['total_chunks'])):.1f}%")
    print("=" * 60)
    
    return all_chunks


# =============================================================
# MAIN - Run this to test
# =============================================================

if __name__ == "__main__":
    # Process all papers
    chunks = process_papers_folder()
    
    if chunks:
        # Show sample chunks
        print("\n" + "=" * 60)
        print("SAMPLE CHUNKS (first 3)")
        print("=" * 60)
        
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata.get('source_file', 'unknown')}")
            print(f"Section: {chunk.metadata.get('section', 'unknown')}")
            print(f"Page: {chunk.metadata.get('page', 'unknown')}")
            print(f"Content preview:\n{chunk.page_content[:300]}...")
        
        # Section distribution
        print("\n" + "=" * 60)
        print("SECTION DISTRIBUTION")
        print("=" * 60)
        
        sections = {}
        for chunk in chunks:
            section = chunk.metadata.get('section', 'unknown')
            sections[section] = sections.get(section, 0) + 1
        
        for section, count in sorted(sections.items(), key=lambda x: -x[1]):
            print(f"  {section}: {count}")