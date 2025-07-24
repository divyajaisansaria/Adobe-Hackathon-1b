# Specify platform for cross-compilation
FROM --platform=linux/amd64 python:3.10-slim

# Set environment variables to force offline mode for Hugging Face libraries AT RUNTIME
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

WORKDIR /app

# Install build tools needed for llama-cpp-python. This is a system dependency.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install all dependencies from the internet. This is the most reliable method.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application script
COPY process_pdf.py .

# Set the default command for the container
CMD ["python", "process_pdf.py", "/app/input", "/app/output"]
