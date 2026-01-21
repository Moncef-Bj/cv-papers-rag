# CV Papers RAG - Multimodal Research Assistant

A **Retrieval-Augmented Generation (RAG)** system for searching and querying Computer Vision research papers. Ask questions in natural language and get AI-synthesized answers with sources and architecture figures.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-orange.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-purple.svg)

---

## Table of Contents

- [Features](#features)
- [Demo](#demo)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Complete Workflow](#complete-workflow)
- [Usage](#usage)
  - [Web Interface](#web-interface)
  - [Command Line Tools](#command-line-tools)
  - [API Endpoints](#api-endpoints)
- [Downloading Papers from arXiv](#downloading-papers-from-arxiv)
- [Maintenance & Debugging](#maintenance--debugging)
- [Configuration](#configuration)
- [Tech Stack](#tech-stack)
- [Performance & Costs](#performance--costs)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## Features

- **Automatic Paper Download**: Fetch papers from arXiv by topic
- **Semantic Search**: Find relevant papers using natural language queries (not just keywords)
- **AI-Powered Answers**: Get synthesized responses from retrieved passages using GPT-4o
- **Architecture Figures**: Search and display architecture diagrams from papers
- **Multimodal Search**: Combine text and figure search for comprehensive results
- **Source Citations**: Every answer includes paper sources with similarity scores
- **Modern Web Interface**: Clean UI with Ask AI and Search modes
- **REST API**: Full API with Swagger documentation
- **Smart Filtering**: Automatically filters out references and related work sections

---

## Demo

### Ask Mode (Web Interface)

Ask a question and get a synthesized answer:

```
Q: "How does person re-identification work with gait features?"

A: "Person re-identification using gait features combines appearance 
   and motion patterns. GAF-Net extracts gait features from skeletal 
   data using Graph Convolutional Networks (GCN) and fuses them with 
   RGB appearance features. This multimodal approach achieves 89.6% 
   Rank-1 accuracy on iLIDS-VID dataset..."

Sources: 
  - VISAPP_2024_157_CR.pdf (68.4% match)
  - algorithms-17-00352-v2.pdf (58.5% match)

Related Figures:
  - [GAF-Net Architecture Diagram]
```

### Command Line Mode

```bash
# Interactive RAG
$ python src/rag.py
RAG System for Computer Vision Papers
Your question: How does attention work in transformers?
ANSWER: [Generated answer with sources]

# Multimodal search
$ python src/multimodal_rag.py
Search: transformer encoder decoder
[Results with text chunks + architecture diagrams]
```

---

## How It Works

### RAG Pipeline

```
+------------------------------------------------------------------+
|                         USER QUESTION                            |
|           "How does person re-identification work?"              |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                    1. RETRIEVAL (Local)                          |
|  +-------------+    +-------------+    +-------------+           |
|  |  Embedding  | -> |  ChromaDB   | -> |   Top-K     |           |
|  |  (MiniLM)   |    |   Search    |    |  Passages   |           |
|  +-------------+    +-------------+    +-------------+           |
|        FREE              FREE             FREE                   |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                    2. GENERATION (LLM)                           |
|  +-------------+    +-------------+    +-------------+           |
|  |  Passages   | -> |   GPT-4o    | -> | Synthesized |           |
|  |  + Question |    |   mini      |    |   Answer    |           |
|  +-------------+    +-------------+    +-------------+           |
|                        ~$0.001/request                           |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|              RESPONSE: Answer + Sources + Figures                |
+------------------------------------------------------------------+
```

### Key Concepts

| Component | What it does | Technology |
|-----------|--------------|------------|
| **Embeddings** | Convert text to vectors for similarity search | all-MiniLM-L6-v2 (local, free) |
| **Vector DB** | Store and search embeddings efficiently | ChromaDB |
| **Retrieval** | Find relevant passages from papers | Cosine similarity |
| **Generation** | Synthesize answer from passages | GPT-4o-mini |
| **Figures** | Extract and describe architecture diagrams | PyMuPDF + GPT-4o Vision |

---

## Project Structure

```
rag-cv-papers/
│
├── src/                           # Core application modules
│   ├── api.py                     # FastAPI REST API (/ask, /search endpoints)
│   ├── arxiv_fetcher.py          # Download papers from arXiv automatically
│   ├── embeddings.py             # Text chunking and indexing into ChromaDB
│   ├── document_processor.py     # PDF processing, intelligent section filtering
│   ├── figure_extractor.py       # Extract architecture figures from PDFs
│   ├── index_figures.py          # Index figure descriptions into ChromaDB
│   ├── rag.py                    # Core RAG pipeline (retrieval + generation)
│   ├── multimodal_rag.py         # Multimodal search (text + figures)
│   └── test_llm.py               # Test OpenAI API connection
│
├── check_desc.py                  # Validate figure descriptions quality
├── debug_figures.py               # Debug ChromaDB figure collection
├── debug_p5.py                    # Debug specific figure extraction issues
├── fix_descriptions.py            # Re-generate invalid figure descriptions
│
├── static/
│   └── index.html                 # Web interface
│
├── data/
│   ├── papers/                    # Your PDF papers (input)
│   ├── figures/                   # Extracted figures (generated)
│   └── figure_descriptions.json   # Figure metadata (generated)
│
├── chroma_db/                     # Vector database (generated)
│   ├── cv_papers/                 # Text chunks collection
│   └── cv_figures/                # Figure descriptions collection
│
├── .env                           # API keys (OPENAI_API_KEY)
├── .env.example                   # Template for environment variables
├── .gitignore                     # Git ignore rules
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- OpenAI API key (for Ask mode and figure descriptions)

### Step 1: Clone the repository

```bash
git clone https://github.com/Moncef-Bj/rag-cv-papers.git
cd rag-cv-papers
```

### Step 2: Create a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure API key

Create a `.env` file (copy from `.env.example`):

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-your-api-key-here
```

### Step 5: Test the setup

```bash
# Test OpenAI connection
python src/test_llm.py
```

If you see "API key loaded successfully" and a response from GPT, you're ready!

---

## Complete Workflow

Here's a **complete example** starting from scratch with "person re-identification" topic:

### 1. Download Papers from arXiv

```bash
# Download 10 papers on person re-identification
python src/arxiv_fetcher.py "person re-identification"

# Or download papers on multiple topics (default behavior)
python src/arxiv_fetcher.py
```

**Output:**
```
Searching arXiv for: person re-identification
Download folder: data/papers
--------------------------------------------------
Deep Learning for Person Re-identification: A Survey...
   Authors: Liang Zheng, Yi Yang, Alexander G. Hauptmann
   Published: 2016-10-13
   Downloaded: 1610.02984_Deep_Learning_for_Person_Re-identification.pdf

... (9 more papers)

==================================================
Fetched 10 papers
Next: Run 'python src/embeddings.py' to index new papers
```

### 2. Index Text Chunks

```bash
python src/embeddings.py
```

**What happens:**
- Loads all PDFs from `data/papers/`
- Extracts text and filters out references, related work
- Chunks documents into ~1000 character pieces
- Generates embeddings (vectors) for each chunk
- Stores everything in ChromaDB

**Output:**
```
============================================================
STEP 1: Loading and chunking PDFs
============================================================
Found 10 PDF files
--------------------------------------------------
Processing: 1610.02984_Deep_Learning_for_Person_Re...
   Pages: 23
   Chunks: 47 -> 31 (filtered 16)
...

PROCESSING SUMMARY
============================================================
Total chunks processed:    458
Chunks kept:               312
Filtered (references):     89
Filtered (related work):   38
Filtered (low quality):    19
Filter rate:               31.9%
============================================================

STEP 2: Creating vector store
============================================================
Vector store created!
   Total vectors: 312
```

### 3. Extract and Describe Figures (Optional but Recommended)

```bash
# Extract architecture diagrams from PDFs
python src/figure_extractor.py
```

**What happens:**
- Scans PDFs for figures (pages 1-10)
- Filters by size, aspect ratio
- Saves to `data/figures/`
- Generates descriptions using GPT-4o Vision
- Saves metadata to `data/figure_descriptions.json`

**Cost:** ~$0.01 per figure

**Output:**
```
============================================================
EXTRACTING ARCHITECTURE FIGURES FROM CV PAPERS
============================================================
Processing: 1610.02984_Deep_Learning_for_Person_Re...
   Found 3 candidate figures
   Saved: 1610.02984_Deep_Learning_p3_img1.png
   Description: "A CNN architecture showing ResNet-50 backbone..."
...

SUMMARY
============================================================
Papers processed: 10
Figures extracted: 27
Descriptions generated: 27
```

### 4. Index Figures for Search

```bash
python src/index_figures.py
```

**Output:**
```
============================================================
 INDEXING FIGURES INTO CHROMADB
============================================================
   Total figures: 27
   Valid descriptions: 25
Indexed 25 figures!
   Collection: cv_figures
```

### 5. Verify Everything Works

```bash
# Check figure descriptions quality
python check_desc.py

# Verify what's in ChromaDB
python debug_figures.py
```

### 6. Start Using!

**Option A: Web Interface**
```bash
python -m uvicorn src.api:app --reload --port 8000
# Open http://localhost:8000
```

**Option B: Command Line RAG**
```bash
python src/rag.py
```

**Option C: Multimodal Search**
```bash
python src/multimodal_rag.py
```

---

## Usage

### Web Interface

1. Start the server:
```bash
python -m uvicorn src.api:app --reload --port 8000
```

2. Open http://localhost:8000 in your browser

3. Choose mode:
   - **Ask AI**: Get synthesized answers with sources
   - **Search**: Get raw search results

4. Example questions:
   - "How does person re-identification work?"
   - "What are the main challenges in multi-camera tracking?"
   - "Explain the difference between CNN and Transformer architectures"
   - "What loss functions are used for metric learning?"

---

### Command Line Tools

#### 1. Interactive RAG (Ask Questions)

```bash
python src/rag.py
```

**Example session:**
```
============================================================
RAG System for Computer Vision Papers
============================================================
Ask questions about the papers in your database.
Type 'quit' to exit.

 Your question: How does person re-identification work?

Retrieving context for: How does person re-identification work?
Context retrieved (3421 chars)
Generating answer...

ANSWER:
Person re-identification (Re-ID) aims to match pedestrian images across 
non-overlapping camera views. Modern approaches use deep learning with 
metric learning objectives. The typical pipeline involves:

1. Feature Extraction: CNN or Transformer backbones extract appearance features
2. Metric Learning: Triplet loss or contrastive loss learn discriminative features
3. Matching: Compute similarity between query and gallery features

According to the papers, state-of-the-art methods achieve 89-95% Rank-1 
accuracy on standard benchmarks like Market-1501 and DukeMTMC-reID.
[Source: 1610.02984_Deep_Learning_for_Person_Re-identification.pdf, Page 4-6]

 Your question: quit
Goodbye!
```

#### 2. Multimodal Search (Text + Figures)

```bash
python src/multimodal_rag.py
```

**Example session:**
```
======================================================================
ARCHITECTURE SEARCH ENGINE
======================================================================
Search for CV architectures using natural language.

Example queries:
  - transformer encoder decoder
  - object detection CNN backbone
  - multi-camera tracking
  - attention mechanism for video

Search: person re-identification CNN architecture

======================================================================
SEARCH: "person re-identification CNN architecture"
======================================================================
   Found: 5 text chunks, 3 figures

----------------------------------------------------------------------
COMBINED RESULTS (sorted by relevance)
----------------------------------------------------------------------

[1] Text | Similarity: 0.842
    Source: 1610.02984_Deep_Learning_for_Person_Re-identificatio...
    Page: 5
    Content: Deep convolutional networks for person Re-ID typically use 
             ResNet or DenseNet backbones. The network learns a metric...

[2] Figure | Similarity: 0.791
    Source: algorithms-17-00352-v2.pdf
    Page: 8
    Title: GAF-Net Architecture Overview
    Description: The architecture shows a dual-branch network with CNN 
                 for appearance and GCN for gait features...
    File: algorithms-17-00352-v2_p8_img2.png

[3] Text | Similarity: 0.756
    Source: VISAPP_2024_157_CR.pdf
    Page: 3
    Content: The proposed architecture uses a ResNet-50 backbone 
             pretrained on ImageNet, followed by a global average...

----------------------------------------------------------------------
ARCHITECTURE FIGURES
----------------------------------------------------------------------

   [1] algorithms-17-00352-v2_p8_img2.png
       Paper: algorithms-17-00352-v2.pdf
       Path: data/figures/algorithms-17-00352-v2_p8_img2.png
       Title: GAF-Net Architecture Overview

   [2] VISAPP_2024_157_CR_p5_img4.png
       Paper: VISAPP_2024_157_CR.pdf
       Path: data/figures/VISAPP_2024_157_CR_p5_img4.png
       Title: Feature Fusion Module

   [3] 1610.02984_Deep_Learning_p7_img1.png
       Paper: 1610.02984_Deep_Learning_for_Person_Re-identification.pdf
       Path: data/figures/1610.02984_Deep_Learning_p7_img1.png
```

#### 3. Download Papers from arXiv

**Single topic:**
```bash
python src/arxiv_fetcher.py "person re-identification"
```

**Multiple topics (default):**
```bash
python src/arxiv_fetcher.py
```

**Suggested Topics for CV Research:**

```python
# Person Re-identification & Tracking
"person re-identification"
"multi-camera tracking"
"pedestrian detection"
"gait recognition"

# Human Pose & Activity
"human pose estimation"
"action recognition"
"3D human reconstruction"

# Object Detection & Segmentation
"object detection transformer"
"instance segmentation"
"semantic segmentation"

# Video Understanding
"video object tracking"
"temporal action detection"
"video captioning"

# 3D Vision
"visual SLAM"
"3D object detection"
"depth estimation"

# Transformers & Attention
"vision transformer"
"self-attention mechanism"
"cross-attention fusion"

# Few-shot & Meta-learning
"few-shot learning"
"meta-learning computer vision"

# Domain Adaptation
"domain adaptation"
"transfer learning"
```

**Custom download example:**
```bash
# Download 20 papers on transformers for CV
python src/arxiv_fetcher.py "vision transformer" --max-results 20
```

---

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/health` | GET | Health check |
| `/stats` | GET | Collection statistics |
| `/search` | GET | Semantic search (retrieval only) |
| `/ask` | GET | Ask AI (RAG with LLM synthesis) |
| `/figures/{name}` | GET | Serve figure images |
| `/docs` | GET | Swagger API documentation |

#### Example API Calls

**Health check:**
```bash
curl http://localhost:8000/health
```

**Get stats:**
```bash
curl http://localhost:8000/stats
```
Response:
```json
{
  "text_chunks": 312,
  "figures": 25,
  "papers": 10
}
```

**Search (retrieval only):**
```bash
curl "http://localhost:8000/search?q=person+re-identification&n_text=5&n_figures=3"
```

**Ask AI (full RAG):**
```bash
curl "http://localhost:8000/ask?q=How+does+person+re-identification+work"
```

Response:
```json
{
  "question": "How does person re-identification work?",
  "answer": "Person re-identification (Re-ID) is the task of matching...",
  "sources": [
    {
      "paper": "1610.02984_Deep_Learning_for_Person_Re-identification.pdf",
      "content": "Person re-identification aims to match pedestrians...",
      "similarity": 0.842,
      "page": 5
    }
  ],
  "figures": [
    {
      "filename": "algorithms-17-00352-v2_p8_img2.png",
      "title": "GAF-Net Architecture",
      "description": "Dual-branch network combining appearance and gait..."
    }
  ]
}
```

---

## Downloading Papers from arXiv

The `arxiv_fetcher.py` tool automatically downloads research papers from arXiv.

### Basic Usage

```bash
# Single topic
python src/arxiv_fetcher.py "person re-identification"

# Multiple topics (default config)
python src/arxiv_fetcher.py
```

### Default Topics

When run without arguments, it downloads papers on these topics:

```
1. person re-identification (15 papers)
2. multi-camera tracking (15 papers)
3. pedestrian detection (15 papers)
4. gait recognition (15 papers)
5. human pose estimation (15 papers)
6. object detection transformer (15 papers)
7. video object tracking (15 papers)
8. action recognition (15 papers)
9. 3D human reconstruction (15 papers)
10. visual SLAM (15 papers)

Total: 150 papers
```

### Customization

Edit `src/arxiv_fetcher.py` to customize:

```python
# Change topics
topics = [
    "your topic 1",
    "your topic 2",
]

# Change papers per topic
papers = fetch_multiple_topics(topics=topics, papers_per_topic=20)
```

### Output

Papers are saved to `data/papers/` with filenames like:
```
2312.12345_Paper_Title_Here.pdf
```

---

## Maintenance & Debugging

### Debug Scripts

#### 1. Check Figure Descriptions Quality

```bash
python check_desc.py
```

Validates that all figure descriptions are valid (not GPT refusals).

**Output:**
```
============================================================
CHECKING VISAPP AND ALGORITHM FIGURES
============================================================

Filename: VISAPP_2024_157_CR_p5_img4.png
Source: VISAPP_2024_157_CR.pdf
Description length: 234 chars
Description: The architecture diagram shows a feature fusion module 
             that combines appearance and gait features using...
STATUS: Description looks OK!
```

#### 2. Debug ChromaDB Contents

```bash
python debug_figures.py
```

Shows what's actually indexed in ChromaDB.

**Output:**
```
============================================================
FIGURES IN CHROMADB
============================================================
  algorithms-17-00352-v2.pdf           -> algorithms-17-00352-v2_p8_img2.png
  VISAPP_2024_157_CR.pdf               -> VISAPP_2024_157_CR_p5_img4.png
  1610.02984_Deep_Learning_for_Per...  -> 1610.02984_Deep_Learning_p7_img1.png

============================================================
SEARCHING FOR VISAPP AND ALGORITHM FIGURES
============================================================

FOUND:
  source_pdf: VISAPP_2024_157_CR.pdf
  filename: VISAPP_2024_157_CR_p5_img4.png
  title: Feature Fusion Architecture
```

#### 3. Debug Specific Figure Issues

```bash
python debug_p5.py
```

Detailed debugging for why a specific figure wasn't extracted/indexed.

#### 4. Fix Invalid Descriptions

```bash
python fix_descriptions.py
```

Re-generates descriptions for figures that have invalid descriptions (GPT refusals).

**Output:**
```
============================================================
RE-DESCRIBING INVALID FIGURES
============================================================

Re-describing: paper_p3_img2.png
  Old description: I cannot analyze this image...
  New description: The diagram shows a ResNet architecture with skip...
  STATUS: Success!

============================================================
Updated 3 figures
Saved to data/figure_descriptions.json

Now run: python src/index_figures.py
```

#### 5. Test LLM Connection

```bash
python src/test_llm.py
```

Verifies that your OpenAI API key works correctly.

---

## Configuration

### Document Processing

The system intelligently filters paper sections:

| Kept | Filtered |
|------|----------|
| Abstract | References |
| Introduction | Related Work |
| Method/Approach | Acknowledgments |
| Architecture | Appendix |
| Experiments | Bibliography |
| Results | |
| Conclusion | |

This filtering reduces noise and improves search quality by ~30%.

### Figure Extraction Settings

In `src/figure_extractor.py`:

```python
MIN_IMAGE_SIZE = 20000      # Minimum 20KB (filters out small icons)
MIN_WIDTH = 300             # Minimum width in pixels
MIN_HEIGHT = 200            # Minimum height in pixels
MIN_ASPECT_RATIO = 0.8      # Accept vertical/square figures
TARGET_PAGES = (1, 10)      # Extract from pages 1-10 only
MAX_FIGURES_PER_PAPER = 3   # Max figures per paper
MAX_TOTAL_FIGURES = 100     # Total limit across all papers
```

### Embedding Settings

In `src/embeddings.py` and `src/index_figures.py`:

```python
USE_OPENAI = False  # Set to True to use OpenAI embeddings (paid)
                    # False = use local all-MiniLM-L6-v2 (free)
```

**Trade-offs:**

| Model | Cost | Quality | Speed |
|-------|------|---------|-------|
| all-MiniLM-L6-v2 (local) | FREE | Good | Fast |
| OpenAI text-embedding-ada-002 | ~$0.10/1M tokens | Better | Slower |

For most use cases, **the free local model is sufficient**.

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Backend** | FastAPI | REST API framework |
| **Vector DB** | ChromaDB | Store and search embeddings |
| **Embeddings** | all-MiniLM-L6-v2 | Convert text to vectors (local, free) |
| **LLM** | OpenAI GPT-4o-mini | Answer synthesis |
| **Vision LLM** | OpenAI GPT-4o | Figure descriptions |
| **PDF Processing** | PyMuPDF (fitz) | Extract text and images |
| **Text Splitting** | LangChain | Chunk documents |
| **arXiv API** | arxiv (Python) | Download papers |
| **Frontend** | HTML/CSS/JavaScript | Web interface |

### Why These Choices?

- **ChromaDB**: Simple, embedded vector DB - no server needed
- **MiniLM**: Free, local embeddings - no API costs for search
- **GPT-4o-mini**: Good quality at low cost (~$0.15/1M tokens)
- **FastAPI**: Modern, fast, automatic API docs
- **PyMuPDF**: Fast PDF processing with image extraction

---

## Performance & Costs

### Performance Metrics

| Metric | Value |
|--------|-------|
| Indexing speed | ~100 papers/minute |
| Search latency | < 100ms |
| Ask latency | 2-5 seconds (LLM dependent) |
| Storage | ~10MB per 100 papers |
| RAM usage | ~500MB with 100 papers indexed |

### Costs Breakdown

| Operation | Cost | Frequency |
|-----------|------|-----------|
| **Text indexing** | FREE | One-time per paper |
| **Search** | FREE | Every search |
| **Ask AI** | ~$0.001 | Per question |
| **Figure extraction** | ~$0.01 | One-time per figure |
| **Figure description** | ~$0.01 | One-time per figure |

**Example Cost for 100 papers:**
- Text indexing: $0 (free, local)
- 30 figures @ $0.01 each: $0.30
- 100 questions @ $0.001 each: $0.10
- **Total: < $1 for complete setup + 100 questions**

### Optimization Tips

1. **Use local embeddings** (already default) - saves 100% on search costs
2. **Limit figure extraction** to key papers - reduce figure description costs
3. **Use GPT-4o-mini** (not GPT-4o) for answers - 10x cheaper
4. **Cache results** - avoid re-asking same questions

---

## Troubleshooting

### "No module named 'chromadb'"

```bash
pip install -r requirements.txt
```

### "OPENAI_API_KEY not found"

Create `.env` file with your API key:

```bash
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

### "No papers found in data/papers/"

Download papers first:

```bash
python src/arxiv_fetcher.py "your topic"
```

Or manually add PDFs to `data/papers/`.

### "Collection 'cv_papers' not found"

Index your papers first:

```bash
python src/embeddings.py
```

### Figures not showing in search results

Run figure extraction and indexing:

```bash
python src/figure_extractor.py
python src/index_figures.py
```

### "Invalid description" errors

Re-generate descriptions:

```bash
python fix_descriptions.py
python src/index_figures.py
```

### ChromaDB is empty after restart

ChromaDB is persistent. If it's empty, you need to re-run:

```bash
python src/embeddings.py
python src/index_figures.py
```

### High OpenAI API costs

1. Set `USE_OPENAI = False` in `embeddings.py` (use free local embeddings)
2. Use GPT-4o-mini instead of GPT-4o
3. Reduce number of retrieved chunks: `TOP_K = 3` in `rag.py`

### Search returns irrelevant results

1. Check if paper sections are properly filtered:
   ```bash
   python src/document_processor.py
   ```

2. Increase chunk overlap for better context:
   ```python
   # In document_processor.py
   chunk_overlap = 300  # Increase from 200
   ```

3. Use more specific queries

---

## Future Improvements

- [ ] **CLIP embeddings** for visual figure search (image-to-image similarity)
- [ ] **Docker containerization** for easy deployment
- [ ] **PDF upload** via web interface (no need for command line)
- [ ] **Citation graph** visualization
- [ ] **Paper summarization** feature
- [ ] **Multi-language support** (French, Chinese, etc.)
- [ ] **Groq/Ollama** as free LLM alternatives
- [ ] **Streamlit dashboard** for advanced analytics
- [ ] **Collaborative annotations** (highlight important passages)
- [ ] **Export to Notion/Obsidian** for note-taking
- [ ] **Automatic paper recommendations** based on your interests
- [ ] **Email digest** of new papers matching your topics

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LangChain](https://langchain.com/) for document processing utilities
- [ChromaDB](https://www.trychroma.com/) for simple and powerful vector storage
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent API framework
- [OpenAI](https://openai.com/) for GPT models and embeddings
- [Sentence Transformers](https://www.sbert.net/) for free local embeddings
- [arXiv](https://arxiv.org/) for open access to research papers

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{boujou2025cvrag,
  author = {Boujou, Moncef},
  title = {CV Papers RAG: Multimodal Research Assistant for Computer Vision},
  year = {2025},
  url = {https://github.com/yourusername/rag-cv-papers}
}
```

---

## Star History

If you find this project useful, please consider giving it a star!

---

**Built with care by a CV researcher, for CV researchers**
