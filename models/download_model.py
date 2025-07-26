# download_models.py
# This script downloads only the essential files for each model to minimize size.

from huggingface_hub import hf_hub_download
import os

# --- Configuration ---
MODELS_DIR = "models"

# Define exactly which files to download for each model repository.
# This gives us precise control over the download size.
MODELS_TO_DOWNLOAD = {
    "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF": {
        "local_dir": "tinyllama-1.1b-chat-v1.0-gguf",
        # UPDATED: Switched to the higher-quality Q4_K_M model.
        "files": ["tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"]
    },
    "BAAI/bge-small-en-v1.5": {
        "local_dir": "bge-small-en-v1.5",
        "files": [
            "config.json",
            "model.safetensors", # Safer and often smaller than pytorch_model.bin
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.txt",
            "modules.json",
            "config_sentence_transformers.json"
        ]
    },
    "cross-encoder/ms-marco-MiniLM-L-6-v2": {
        "local_dir": "cross-encoder-ms-marco",
        "files": [
            "config.json",
            "model.safetensors",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.txt"
        ]
    }
}

def download_all_models():
    """
    Iterates through the model dictionary and downloads each required file individually.
    """
    print("--- Starting Model Download Process ---")
    os.makedirs(MODELS_DIR, exist_ok=True)

    for repo_id, info in MODELS_TO_DOWNLOAD.items():
        local_dir_path = os.path.join(MODELS_DIR, info["local_dir"])
        print(f"\nProcessing repository: {repo_id}")

        for filename in info["files"]:
            local_filepath = os.path.join(local_dir_path, filename)
            
            if os.path.exists(local_filepath):
                print(f"  ✅ File '{filename}' already exists. Skipping.")
                continue
            
            print(f"  📥 Downloading '{filename}'...")
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=local_dir_path,
                    local_dir_use_symlinks=False # To suppress the warning
                )
            except Exception as e:
                print(f"  ❌ FAILED to download '{filename}'. Error: {e}")

    print("\n--- Model Download Process Complete ---")

if __name__ == "__main__":
    download_all_models()
