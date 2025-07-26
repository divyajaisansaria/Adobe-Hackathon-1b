# 📄 Adobe PDF Content Extractor (Hackathon-1B)

This project uses a containerized **Docker** environment to run an advanced model for extracting structured data from PDF documents. This version requires **Git LFS** (Large File Storage) to handle the large model files.

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

The script expects a specific folder structure inside the `input` directory. Place your collection folders inside it like this:

```bash
input/
├── Collection 1/
│   ├── PDFs/
│   │   └── (Your PDF files for collection 1)
│   └── challenge1b_input.json
│
└── Collection 2/
├── PDFs/
│   └── (Your PDF files for collection 2)
└── challenge1b_input.json
```

### Step 4: Build and Run the Docker Container

**Build the Docker Image:**
This command builds the image, copying the code and model files into it.
- **On Windows**
```bash
docker build -t adobe-round-1b .
docker run --rm -v ${PWD}/input:/app/input:ro -v ${PWD}/output/adobe-round-1b/:/app/output --network none adobe-round-1b

```

- **On macOS / Linux:**
```bash
docker build -t adobe-round-1b .
docker run --rm -v $(pwd)/input:/app/input:ro -v $(pwd)/output/adobe-round-1b/:/app/output --network none adobe-round-1b

```

### Step 5: Check the Output

The resulting `.json` files will be saved in the `output/adobe-round-1b/` directory on your computer.

## 🧠 Project Explanation

### File Structure
* **`Dockerfile`**: Builds the container, installing all dependencies and "baking in" the model.
* **`process_pdf.py`**: The main Python script that orchestrates the PDF processing.
* **`model/`**: Contains the large ML model files, managed by Git LFS.
* **`input/`**: Directory where you place your source `Collection` folders.
* **`output/`**: Directory where the final JSON results are written.

### How It Works
The `docker build` command creates a self-contained image with the Python environment, all dependencies, and the large model files. When you execute `docker run`, the container starts and runs the `process_pdf.py` script. The script reads the PDF collections from the mounted `/app/input` folder, uses the internal model to perform data extraction, and writes the structured JSON output to the `/app/output` folder, which appears on your local machine.
