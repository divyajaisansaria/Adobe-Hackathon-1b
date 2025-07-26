# 📄 Adobe Hackathon 1B – Persona-Driven PDF Content Extractor

This project uses a containerized **Docker** environment to run an advanced AI-powered pipeline that extracts and ranks **relevant sections and sub-sections** from multiple PDF documents based on a given **persona** and their **task (job-to-be-done)**.

It leverages **Git LFS** (Large File Storage) to manage lightweight language models and embedding tools required for offline semantic analysis.

## 🚀 How to Run

Follow these steps to set up and run the project on your local machine.

### Step 1: Install Prerequisites

You need **Docker Desktop** installed on your system.

#### 🐳 Docker Desktop
* Ensure Docker Desktop is installed and running.
* **[Download Docker Desktop](https://www.docker.com/products/docker-desktop/)**

### Step 2: Clone the Repository with LFS

Open your terminal (PowerShell or Git Bash on Windows) and run the following commands.

- **On Windows**
```bash
git lfs install
git clone https://github.com/divyajaisansaria/Adobe-Hackathon-1b
cd Adobe-Hackathon-1b
    
```

- **On macOS / Linux:**
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install git-lfs
git lfs install
git clone https://github.com/divyajaisansaria/Adobe-Hackathon-1b
cd Adobe-Hackathon-1b
    
```   

### Step 3: Prepare Input Data

The script expects a specific folder structure inside the `input` directory. Place your input files inside it like this:

```bash
input/
├── PDFs/
│   │   └── (Your PDF files)
│   └──input.json
```

### Step 4: Build and Run the Docker Container

**Build and Run the Docker Image:**
These commands builds the image, copying the code and model files into it, and runs the image.
- **On Windows**
```bash
docker build -t adobe-round-1b .

```
```bash
docker run --rm -v ${PWD}/input:/app/input:ro -v ${PWD}/output/adobe-round-1b/:/app/output --network none adobe-round-1b

```

- **On macOS / Linux:**
```bash
docker build -t adobe-round-1b .

```
```bash
docker run --rm -v $(pwd)/input:/app/input:ro -v $(pwd)/output/adobe-round-1b/:/app/output --network none adobe-round-1b

```

### Step 5: Check the Output

The resulting `.json` files will be saved in the `output/adobe-round-1b/` directory on your computer.

## 🧠 Project Explanation
This project solves the Adobe Hackathon Round 1B challenge by building a modular, offline, CPU-only pipeline that intelligently extracts **persona-relevant content** from diverse PDF documents. The pipeline combines layout-based PDF parsing, language model-driven query generation, multi-query semantic search, and cross-encoder reranking for high-precision content selection.

### 🧩 Step-by-Step Breakdown

1. **PDF Structure Parsing**
   - Uses `pdfplumber` to extract all lines from each PDF, along with visual features like:
     - Font size
     - Boldness
     - Word count
     - Vertical spacing
   - Headings are identified using a scoring function, and documents are chunked into sections based on heading boundaries.

2. **Heading Classification & Chunking**
   - Headings are classified into H1, H2, or H3 levels using regex patterns and visual cues.
   - Each heading and its following body content is grouped as a **chunk**, tagged with page number and document ID.

3. **Persona Understanding**
   - The given persona and job-to-be-done are processed by a lightweight **TinyLLaMA model** (running locally via `llama-cpp`) to generate **5 diverse semantic queries**.
   - These queries simulate how the persona would search the documents to complete their task.

4. **Multi-Query Embedding & Ranking**
   - Queries and all section chunks are embedded using the **`bge-small-en-v1.5`** model (via `sentence-transformers`).
   - Cosine similarity is calculated between queries and chunks.
   - Scores are **weighted across queries**, and the most relevant chunks are shortlisted.
   - A **document diversity cap** (max 2 chunks per PDF) ensures variety.

5. **Sub-Section Refinement**
   - From the top-ranked chunks, a cross-encoder (`cross-encoder/ms-marco`) is used to **rerank** based on deeper semantic alignment.
   - The top 5 final selections are returned with their document name, page number, section title, and a short text extract.

### 🧭 Pipeline Flow Diagram

```mermaid
graph TD
  A[Start: Collection Input] --> B[PDF Parsing via pdfplumber]
  B --> C[Heading Detection & Chunking]
  C --> D[Persona & Task from challenge1b_input.json]
  D --> E[TinyLLaMA generates 5 semantic queries]
  C --> F[Embed chunks using BGE model]
  E --> G[Embed queries using BGE model]
  F --> H[Cosine similarity scoring (multi-query)]
  G --> H
  H --> I[Top chunks (max 2 per document)]
  I --> J[Cross-Encoder Re-ranking]
  J --> K[Top 5 Results: section title + page + document + refined text]
  K --> L[Write JSON to /app/output/]

---


### File Structure
* **`Dockerfile`**: Builds the container, installing all dependencies and "baking in" the model.
* **`process_pdf.py`**: The main Python script that orchestrates the PDF processing.
* **`model/`**: Contains the large ML model files, managed by Git LFS.
* **`input/`**: Directory where you place your input pdf files and input.json.
* **`output/`**: Directory where the final JSON results are written.
  
### ✅ Key Highlights
- 🔒 **Fully Offline**: No internet access required at any stage.
- ⚡ **Fast Execution**: ≤ 60 seconds for 7-10 PDFs on CPU.
- 🧠 **Multi-angle Relevance**: Multiple persona queries + diversity-aware ranking.
- 🔍 **Granular Extraction**: Extracts section-level and sub-paragraph insights.
- 📚 **Reusable Modular Code**: Clean functions for chunking, embedding, and reranking.


### How It Works
The `docker build` command creates a self-contained image with the Python environment, all dependencies, and the large model files. When you execute `docker run`, the container starts and runs the `process_pdf.py` script. The script reads the PDF collections from the mounted `/app/input` folder, uses the internal model to perform data extraction, and writes the structured JSON output to the `/app/output` folder, which appears on your local machine.
