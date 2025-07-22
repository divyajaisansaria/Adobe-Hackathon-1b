#!/bin/bash

echo "🔧 Creating folders..."
mkdir -p models

echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "📥 Downloading TinyLlama model..."
wget -O models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  https://huggingface.co/cmp-nct/tinyllama-1.1b-chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

echo "📥 Downloading BGE-small-en-v1.5 model from Hugging Face..."
huggingface-cli download BAAI/bge-small-en-v1.5 --local-dir models/bge-small-en-v1.5 --local-dir-use-symlinks False

echo "✅ Setup complete. You're ready to run the pipeline!"
