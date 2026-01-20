"""
Figure Extractor for Architecture Search Engine
Extracts and selects ARCHITECTURE DIAGRAMS from CV papers.

FEATURES:
- Detects new papers that don't have figures extracted yet
- Only processes new papers (incremental extraction)
- Keeps existing figures and descriptions
- Combines old + new for final selection
"""

import fitz  # PyMuPDF
import base64
import os
import json
import re
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# =============================================================
# CONFIGURATION
# =============================================================

FIGURES_DIR = Path("data/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PAPERS_DIR = Path("data/papers")
CACHE_FILE = Path("data/figure_descriptions.json")

# Extraction filters
MIN_IMAGE_SIZE = 20000   # Minimum bytes (20KB instead of 50KB)
MIN_WIDTH = 300          # Minimum width in pixels
MIN_HEIGHT = 200         # Minimum height in pixels

# Selection filters
TARGET_PAGES = (1, 10)          # Pages 1-10 (more inclusive)
MIN_ASPECT_RATIO = 0.8         # Slightly more inclusive
MAX_FIGURES_PER_PAGE = 3        # Skip pages with many figures
MAX_FIGURES_PER_PAPER = 3       # More figures per paper
MAX_TOTAL_FIGURES = 500         # Higher budget

# Keywords for title filtering
KEEP_KEYWORDS = [
    "architecture", "pipeline", "framework", "overview", "network",
    "model", "module", "encoder", "decoder", "block", "structure",
    "method", "approach", "system", "flow", "proposed", "diagram",
    "scheme", "design", "workflow", "gait", "fusion", "feature"
]

SKIP_KEYWORDS = [
    "results", "comparison", "qualitative", "ablation", "samples",
    "visualization", "examples", "dataset", "failure", "user study",
    "quantitative", "evaluation", "benchmark", "generated",
    "output", "ground truth", "baseline", "curve", "plot", "graph"
]


# =============================================================
# HELPER FUNCTIONS
# =============================================================

def get_paper_id(pdf_name: str) -> str:
    """Get a consistent paper ID from PDF name."""
    return Path(pdf_name).stem[:25]


def extract_figure_title(pdf_path: str, page_num: int, img_index: int) -> str:
    """Extract the caption/title of a figure from the PDF."""
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        text = page.get_text()
        doc.close()
        
        patterns = [
            r'Figure\s*(\d+)[.:]\s*([^\n]+)',
            r'Fig\.\s*(\d+)[.:]\s*([^\n]+)',
            r'FIGURE\s*(\d+)[.:]\s*([^\n]+)',
        ]
        
        titles = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                fig_num, title = match
                titles.append({
                    "fig_num": int(fig_num),
                    "title": title.strip()[:200]
                })
        
        if titles:
            return titles[0]["title"]
        return ""
        
    except Exception:
        return ""


def is_architecture_title(title: str) -> bool:
    """Check if the figure title suggests it's an architecture diagram."""
    if not title:
        return True
    
    title_lower = title.lower()
    
    for kw in SKIP_KEYWORDS:
        if kw in title_lower:
            return False
    
    for kw in KEEP_KEYWORDS:
        if kw in title_lower:
            return True
    
    return True


# =============================================================
# DETECT NEW PAPERS
# =============================================================

def get_papers_with_figures() -> set:
    """Get set of paper IDs that already have figures extracted."""
    existing_papers = set()
    
    for img_path in FIGURES_DIR.glob("*.*"):
        if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
            continue
        
        filename = img_path.name
        match = re.match(r'(.+)_p(\d+)_img(\d+)\.(\w+)', filename)
        if match:
            paper_id = match.group(1)
            existing_papers.add(paper_id)
    
    return existing_papers


def get_all_papers() -> list:
    """Get all PDF files in papers directory."""
    return list(PAPERS_DIR.glob("*.pdf"))


def get_new_papers() -> list:
    """Get papers that don't have figures extracted yet."""
    existing_paper_ids = get_papers_with_figures()
    all_pdfs = get_all_papers()
    
    new_papers = []
    for pdf_path in all_pdfs:
        paper_id = get_paper_id(pdf_path.name)
        if paper_id not in existing_paper_ids:
            new_papers.append(pdf_path)
    
    return new_papers


# =============================================================
# EXTRACT IMAGES FROM PDF
# =============================================================

def extract_images_from_pdf(pdf_path: str) -> list[dict]:
    """Extract all images from a PDF file with metadata."""
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    
    images = []
    figures_per_page = {}
    
    # First pass: count figures per page
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images()
        valid_count = 0
        for img in image_list:
            try:
                base_image = doc.extract_image(img[0])
                if (len(base_image["image"]) >= MIN_IMAGE_SIZE and
                    base_image["width"] >= MIN_WIDTH and
                    base_image["height"] >= MIN_HEIGHT):
                    valid_count += 1
            except:
                pass
        figures_per_page[page_num] = valid_count
    
    # Second pass: extract images
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                width = base_image["width"]
                height = base_image["height"]
                
                if len(image_bytes) < MIN_IMAGE_SIZE:
                    continue
                if width < MIN_WIDTH or height < MIN_HEIGHT:
                    continue
                
                aspect_ratio = width / height if height > 0 else 0
                
                paper_id = get_paper_id(pdf_path.name)
                filename = f"{paper_id}_p{page_num+1}_img{img_index+1}.{image_ext}"
                filepath = FIGURES_DIR / filename
                
                # Save image
                with open(filepath, "wb") as f:
                    f.write(image_bytes)
                
                title = extract_figure_title(str(pdf_path), page_num, img_index)
                
                images.append({
                    "filepath": str(filepath),
                    "filename": filename,
                    "source_pdf": pdf_path.name,
                    "page": page_num + 1,
                    "width": width,
                    "height": height,
                    "aspect_ratio": round(aspect_ratio, 2),
                    "size_bytes": len(image_bytes),
                    "figures_on_page": figures_per_page.get(page_num, 1),
                    "title": title,
                })
                
            except Exception:
                pass
    
    doc.close()
    return images


# =============================================================
# LOAD EXISTING DATA
# =============================================================

def load_existing_figures() -> list[dict]:
    """Load metadata from already extracted figures."""
    if not FIGURES_DIR.exists():
        return []
    
    print(f"   Loading existing figures from {FIGURES_DIR}...")
    
    # First, build a mapping from paper_id to actual PDF filename
    pdf_mapping = {}
    for pdf_path in PAPERS_DIR.glob("*.pdf"):
        paper_id = get_paper_id(pdf_path.name)
        pdf_mapping[paper_id] = pdf_path.name
        # Also map the full stem (without .pdf) for non-truncated names
        full_stem = pdf_path.stem
        pdf_mapping[full_stem] = pdf_path.name
    
    print(f"   Found {len(pdf_mapping)//2} PDFs in papers directory")
    
    figures_by_pdf = {}
    
    for img_path in FIGURES_DIR.glob("*.*"):
        if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
            continue
        
        filename = img_path.name
        match = re.match(r'(.+)_p(\d+)_img(\d+)\.(\w+)', filename)
        if not match:
            continue
        
        paper_id, page, img_idx, ext = match.groups()
        
        try:
            from PIL import Image
            with Image.open(img_path) as img:
                width, height = img.size
        except:
            continue
        
        aspect_ratio = width / height if height > 0 else 0
        
        # Find the actual PDF name
        actual_pdf = None
        if paper_id in pdf_mapping:
            actual_pdf = pdf_mapping[paper_id]
        else:
            # Try to find a matching PDF by prefix
            for pdf_id, pdf_name in pdf_mapping.items():
                if pdf_name.startswith(paper_id) or paper_id.startswith(pdf_id.replace('.pdf', '')):
                    actual_pdf = pdf_name
                    break
        
        if not actual_pdf:
            # Use paper_id + .pdf as fallback
            actual_pdf = f"{paper_id}.pdf"
        
        fig_data = {
            "filepath": str(img_path),
            "filename": filename,
            "source_pdf": actual_pdf,
            "page": int(page),
            "img_index": int(img_idx),
            "width": width,
            "height": height,
            "aspect_ratio": round(aspect_ratio, 2),
            "size_bytes": img_path.stat().st_size,
            "title": "",
        }
        
        if paper_id not in figures_by_pdf:
            figures_by_pdf[paper_id] = []
        figures_by_pdf[paper_id].append(fig_data)
    
    # Get titles from PDFs
    for pdf_path in PAPERS_DIR.glob("*.pdf"):
        paper_id = get_paper_id(pdf_path.name)
        full_stem = pdf_path.stem
        
        # Check both truncated and full paper_id
        matching_key = None
        if paper_id in figures_by_pdf:
            matching_key = paper_id
        elif full_stem in figures_by_pdf:
            matching_key = full_stem
        else:
            # Try prefix matching
            for key in figures_by_pdf.keys():
                if key.startswith(paper_id) or paper_id.startswith(key):
                    matching_key = key
                    break
                if key.startswith(full_stem) or full_stem.startswith(key):
                    matching_key = key
                    break
        
        if not matching_key:
            continue
        
        for fig in figures_by_pdf[matching_key]:
            title = extract_figure_title(
                str(pdf_path), 
                fig["page"] - 1,
                fig["img_index"]
            )
            fig["title"] = title
            fig["source_pdf"] = pdf_path.name
    
    # Flatten and count figures per page
    all_figures = []
    for paper_id, figures in figures_by_pdf.items():
        page_counts = {}
        for fig in figures:
            page = fig["page"]
            page_counts[page] = page_counts.get(page, 0) + 1
        
        for fig in figures:
            fig["figures_on_page"] = page_counts.get(fig["page"], 1)
            all_figures.append(fig)
    
    print(f"   Found {len(all_figures)} existing figures")
    return all_figures


def load_cached_descriptions() -> dict:
    """Load previously generated descriptions."""
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'r') as f:
            figures = json.load(f)
        # Create lookup by filename
        return {f['filename']: f.get('description', '') for f in figures}
    return {}


# =============================================================
# SMART FIGURE SELECTION
# =============================================================

def select_architecture_figures(all_figures: list[dict]) -> list[dict]:
    """Select figures that are likely ARCHITECTURE DIAGRAMS."""
    
    print(f"\n{'='*60}")
    print(" SELECTING ARCHITECTURE FIGURES")
    print("="*60)
    print("Filters:")
    print(f"   - Pages: {TARGET_PAGES[0]}-{TARGET_PAGES[1]}")
    print(f"   - Aspect ratio: > {MIN_ASPECT_RATIO}")
    print(f"   - Max figures per page: {MAX_FIGURES_PER_PAGE}")
    print(f"   - Max per paper: {MAX_FIGURES_PER_PAPER}")
    print(f"   - Max total: {MAX_TOTAL_FIGURES}")
    print("="*60)
    
    by_paper = {}
    for fig in all_figures:
        paper = fig['source_pdf']
        if paper not in by_paper:
            by_paper[paper] = []
        by_paper[paper].append(fig)
    
    selected = []
    stats = {
        "total": len(all_figures),
        "skip_page": 0,
        "skip_ratio": 0,
        "skip_crowded": 0,
        "skip_title": 0,
        "selected": 0,
    }
    
    for paper_name, figures in by_paper.items():
        candidates = []
        
        for fig in figures:
            page = fig['page']
            aspect_ratio = fig['aspect_ratio']
            figures_on_page = fig.get('figures_on_page', 1)
            title = fig.get('title', '')
            
            if page < TARGET_PAGES[0] or page > TARGET_PAGES[1]:
                stats["skip_page"] += 1
                continue
            
            if aspect_ratio < MIN_ASPECT_RATIO:
                stats["skip_ratio"] += 1
                continue
            
            if figures_on_page > MAX_FIGURES_PER_PAGE:
                stats["skip_crowded"] += 1
                continue
            
            if not is_architecture_title(title):
                stats["skip_title"] += 1
                continue
            
            # Score for ranking
            score = 0
            score += (11 - page) * 10  # Earlier page = higher
            score += aspect_ratio * 5
            score += 20 if any(kw in title.lower() for kw in KEEP_KEYWORDS) else 0
            
            fig['_score'] = score
            candidates.append(fig)
        
        candidates.sort(key=lambda f: f['_score'], reverse=True)
        
        for fig in candidates[:MAX_FIGURES_PER_PAPER]:
            if len(selected) >= MAX_TOTAL_FIGURES:
                break
            
            selected.append(fig)
            stats["selected"] += 1
            
            title_preview = fig.get('title', '')[:40] + "..." if fig.get('title') else "(no title)"
            print(f"   {fig['filename']}")
            print(f"      Page {fig['page']}, Ratio {fig['aspect_ratio']}, Score {fig['_score']:.0f}")
            print(f"      Title: {title_preview}")
        
        if len(selected) >= MAX_TOTAL_FIGURES:
            break
    
    print(f"\n{'='*60}")
    print(" SELECTION SUMMARY")
    print("="*60)
    print(f"   Total figures: {stats['total']}")
    print(f"   Skipped (wrong page): {stats['skip_page']}")
    print(f"   Skipped (aspect ratio): {stats['skip_ratio']}")
    print(f"   Skipped (crowded page): {stats['skip_crowded']}")
    print(f"   Skipped (title keywords): {stats['skip_title']}")
    print(f"   Selected: {stats['selected']}")
    print(f"   Papers represented: {len(set(f['source_pdf'] for f in selected))}")
    
    return selected


# =============================================================
# DESCRIBE FIGURES WITH VISION LLM
# =============================================================

def encode_image_base64(image_path: str) -> str:
    """Encode image to base64 for API."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def describe_figure(image_path: str) -> str:
    """Use GPT-4o to describe a figure from a CV paper."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    base64_image = encode_image_base64(image_path)
    
    ext = Path(image_path).suffix.lower()
    media_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(ext, "image/png")
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a computer vision expert. Describe technical architecture diagrams from research papers. Be specific about components, layers, and data flow. Always provide a description."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Describe this architecture diagram from a computer vision paper in 2-3 sentences.

Focus on:
- Main components (encoder, decoder, backbone, head, etc.)
- Type of architecture (CNN, Transformer, U-Net, etc.)
- Data flow and connections between components
- Any visible labels or module names

Be direct and technical."""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{base64_image}",
                            "detail": "low"
                        }
                    }
                ]
            }
        ],
        max_tokens=200
    )
    
    return response.choices[0].message.content


def describe_figures_incremental(figures: list[dict], cached_descriptions: dict) -> list[dict]:
    """
    Describe figures, reusing cached descriptions when available.
    Only calls API for figures without descriptions.
    """
    to_describe = []
    already_described = []
    
    for fig in figures:
        filename = fig['filename']
        if filename in cached_descriptions and cached_descriptions[filename]:
            fig['description'] = cached_descriptions[filename]
            already_described.append(fig)
        else:
            to_describe.append(fig)
    
    print(f"\n{'='*60}")
    print(f" DESCRIBING FIGURES WITH GPT-4o")
    print("="*60)
    print(f"   Already described (cached): {len(already_described)}")
    print(f"   Need to describe: {len(to_describe)}")
    
    if to_describe:
        print(f"\n   Describing {len(to_describe)} new figures...")
        
        for i, fig in enumerate(to_describe):
            print(f"\n   [{i+1}/{len(to_describe)}] {fig['filename']}")
            
            try:
                description = describe_figure(fig['filepath'])
                fig['description'] = description
                print(f"      OK ({len(description)} chars)")
            except Exception as e:
                print(f"      Error: {e}")
                fig['description'] = None
    
    all_figures = already_described + to_describe
    described = sum(1 for fig in all_figures if fig.get('description'))
    print(f"\n   Total described: {described}/{len(all_figures)}")
    
    return all_figures


# =============================================================
# VALIDATION
# =============================================================

def is_valid_description(desc: str) -> bool:
    """Check if description is valid (not a refusal from GPT)."""
    if not desc or len(desc) < 50:
        return False
    
    bad_phrases = [
        "unable to analyze", "can't analyze", "cannot analyze",
        "i'm unable", "i cannot", "provide a description",
        "provide more context", "describe it or provide",
        "i can't directly", "cannot directly", "if you describe it",
        "i'm not able", "cannot view", "can't view",
    ]
    
    desc_lower = desc.lower()
    return not any(phrase in desc_lower for phrase in bad_phrases)


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":
    print("="*60)
    print("  ARCHITECTURE FIGURE EXTRACTOR (Incremental)")
    print("="*60)
    
    # Step 1: Check for new papers
    print("\n STEP 1: Checking for new papers")
    print("-"*60)
    
    new_papers = get_new_papers()
    all_papers = get_all_papers()
    existing_paper_ids = get_papers_with_figures()
    
    print(f"   Total papers: {len(all_papers)}")
    print(f"   Papers with figures: {len(existing_paper_ids)}")
    print(f"   New papers to process: {len(new_papers)}")
    
    if new_papers:
        print("\n   New papers found:")
        for p in new_papers:
            print(f"      - {p.name}")
    
    # Step 2: Extract figures from new papers
    if new_papers:
        print("\n STEP 2: Extracting figures from new papers")
        print("-"*60)
        
        for i, pdf_path in enumerate(new_papers):
            print(f"\n   [{i+1}/{len(new_papers)}] {pdf_path.name[:50]}...")
            images = extract_images_from_pdf(pdf_path)
            print(f"      Extracted {len(images)} figures")
    else:
        print("\n STEP 2: No new papers to extract")
    
    # Step 3: Load all figures (existing + new)
    print("\n STEP 3: Loading all figures")
    print("-"*60)
    
    all_figures = load_existing_figures()
    
    if not all_figures:
        print("   No figures found!")
        exit()
    
    print(f"   Total figures available: {len(all_figures)}")
    
    # Step 4: Smart selection
    print("\n STEP 4: Selecting architecture figures")
    print("-"*60)
    
    selected_figures = select_architecture_figures(all_figures)
    
    if not selected_figures:
        print("   No figures selected!")
        exit()
    
    # Step 5: Describe with GPT-4o (incremental)
    print("\n STEP 5: Describing figures")
    print("-"*60)
    
    cached_descriptions = load_cached_descriptions()
    described_figures = describe_figures_incremental(selected_figures, cached_descriptions)
    
    # Step 6: Save
    with open(CACHE_FILE, 'w') as f:
        json.dump(described_figures, f, indent=2)
    print(f"\n   Saved to {CACHE_FILE}")
    
    # Final summary
    valid_count = sum(1 for f in described_figures 
                      if f.get('description') and is_valid_description(f['description']))
    
    print(f"\n{'='*60}")
    print(" FINAL SUMMARY")
    print("="*60)
    print(f"   Total figures available: {len(all_figures)}")
    print(f"   Selected: {len(selected_figures)}")
    print(f"   Described: {sum(1 for f in described_figures if f.get('description'))}")
    print(f"   Valid descriptions: {valid_count}")
    
    new_described = sum(1 for f in described_figures 
                       if f['filename'] not in cached_descriptions and f.get('description'))
    if new_described > 0:
        print(f"\n   NEW descriptions added: {new_described}")
        print(f"   Estimated cost: ~${new_described * 0.01:.2f}")
    
    # Show new papers' figures
    if new_papers:
        print(f"\n{'='*60}")
        print(" FIGURES FROM NEW PAPERS")
        print("="*60)
        
        new_paper_names = {p.name for p in new_papers}
        for fig in described_figures:
            if fig['source_pdf'] in new_paper_names:
                print(f"\n   {fig['filename']}")
                print(f"      Paper: {fig['source_pdf'][:50]}")
                print(f"      Title: {fig.get('title', 'N/A')[:60]}")
                if fig.get('description'):
                    print(f"      Desc: {fig['description'][:100]}...")