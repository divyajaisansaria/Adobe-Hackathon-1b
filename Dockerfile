FROM --platform=linux/amd64 python:3.10-slim
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY process_pdf.py .
COPY models/ /app/models/
CMD ["python", "process_pdf.py", "/app/input", "/app/output", "/app/models"]
