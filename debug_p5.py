"""
Debug script to understand why VISAPP p5 figure is not in the JSON
"""
import json
import re
from pathlib import Path

FIGURES_DIR = Path("data/figures")
PAPERS_DIR = Path("data/papers")
JSON_FILE = Path("data/figure_descriptions.json")

# Config from figure_extractor.py
MIN_IMAGE_SIZE = 20000
MIN_WIDTH = 300
MIN_HEIGHT = 200
TARGET_PAGES = (1, 10)
MIN_ASPECT_RATIO = 1.2
MAX_FIGURES_PER_PAGE = 3
MAX_FIGURES_PER_PAPER = 3
MAX_TOTAL_FIGURES = 100

print("="*60)
print("DEBUG: Why is VISAPP p5 not in JSON?")
print("="*60)

# Step 1: Check what's in data/figures/
print("\n[STEP 1] Files in data/figures/ with VISAPP:")
visapp_files = list(FIGURES_DIR.glob("*VISAPP*"))
for f in visapp_files:
    print(f"  - {f.name} ({f.stat().st_size} bytes)")

# Step 2: Check what's in JSON
print("\n[STEP 2] VISAPP entries in JSON:")
with open(JSON_FILE, 'r') as f:
    figures_json = json.load(f)

visapp_in_json = [fig for fig in figures_json if 'VISAPP' in fig.get('filename', '')]
for fig in visapp_in_json:
    print(f"  - {fig.get('filename')}")
    print(f"    Page: {fig.get('page')}, Desc: {fig.get('description', '')[:50]}...")

if not visapp_in_json:
    print("  (none)")

# Step 3: Simulate load_existing_figures() for VISAPP
print("\n[STEP 3] Simulating load_existing_figures() for VISAPP files:")
for img_path in visapp_files:
    filename = img_path.name
    print(f"\n  Checking: {filename}")
    
    # Parse filename
    match = re.match(r'(.+)_p(\d+)_img(\d+)\.(\w+)', filename)
    if not match:
        print(f"    PROBLEM: Filename doesn't match pattern!")
        continue
    
    paper_id, page, img_idx, ext = match.groups()
    print(f"    Parsed: paper_id={paper_id}, page={page}, img_idx={img_idx}")
    
    # Get image dimensions
    try:
        from PIL import Image
        with Image.open(img_path) as img:
            width, height = img.size
        print(f"    Dimensions: {width}x{height}")
        
        aspect_ratio = width / height
        print(f"    Aspect ratio: {aspect_ratio:.2f}")
        
        size_bytes = img_path.stat().st_size
        print(f"    Size: {size_bytes} bytes")
        
        # Check filters
        print(f"\n    FILTER CHECKS:")
        print(f"    - Page {page} in range {TARGET_PAGES}? {TARGET_PAGES[0] <= int(page) <= TARGET_PAGES[1]}")
        print(f"    - Aspect ratio {aspect_ratio:.2f} >= {MIN_ASPECT_RATIO}? {aspect_ratio >= MIN_ASPECT_RATIO}")
        print(f"    - Size {size_bytes} >= {MIN_IMAGE_SIZE}? {size_bytes >= MIN_IMAGE_SIZE}")
        print(f"    - Width {width} >= {MIN_WIDTH}? {width >= MIN_WIDTH}")
        print(f"    - Height {height} >= {MIN_HEIGHT}? {height >= MIN_HEIGHT}")
        
    except Exception as e:
        print(f"    ERROR: {e}")

# Step 4: Check how many figures are in JSON total
print(f"\n[STEP 4] Total figures in JSON: {len(figures_json)}")
print(f"  MAX_TOTAL_FIGURES setting: {MAX_TOTAL_FIGURES}")

# Step 5: Check if VISAPP was processed at all
print("\n[STEP 5] Looking for any VISAPP-related entries by source_pdf:")
for fig in figures_json:
    source = fig.get('source_pdf', '')
    if 'VISAPP' in source.upper():
        print(f"  Found: {fig.get('filename')} from {source}")

print("\n" + "="*60)
print("CONCLUSION:")
print("="*60)

# Determine the issue
p5_in_files = any('p5' in f.name for f in visapp_files)
p5_in_json = any('p5' in fig.get('filename', '') for fig in visapp_in_json)

if p5_in_files and not p5_in_json:
    print("p5 EXISTS in data/figures/ but NOT in JSON!")
    print("\nPossible causes:")
    print("1. load_existing_figures() failed to detect it")
    print("2. It was filtered during select_architecture_figures()")
    print("3. MAX_TOTAL_FIGURES was reached before p5 was added")
    print("4. The figure was extracted AFTER the last figure_extractor.py run")