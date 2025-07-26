from huggingface_hub import hf_hub_download, HfApi
import os

MODELS_DIR = "models"
MODELS_TO_DOWNLOAD = {
    "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF": {
        "local_dir": "tinyllama-1.1b-chat-v1.0-gguf",
        "files": ["tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"]
    },
    "BAAI/bge-small-en-v1.5": {
        "local_dir": "bge-small-en-v1.5",
        "files": [
            "config.json", "model.safetensors", "special_tokens_map.json",
            "tokenizer_config.json", "tokenizer.json", "vocab.txt",
            "modules.json", "config_sentence_transformers.json"
        ]
    },
    "cross-encoder/ms-marco-MiniLM-L-6-v2": {
        "local_dir": "cross-encoder-ms-marco",
        "files": [
            "config.json", "model.safetensors", "special_tokens_map.json",
            "tokenizer_config.json", "tokenizer.json", "vocab.txt"
        ]
    }
}

def format_size(size_bytes):
    if size_bytes is None:
        return "Size not available"
    if size_bytes >= 1024**3:
        return f"{size_bytes / 1024**3:.2f} GB"
    if size_bytes >= 1024**2:
        return f"{size_bytes / 1024**2:.2f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"
    return f"{size_bytes} Bytes"

def download_all_models():
    print("--- Starting Model Download Process ---")
    os.makedirs(MODELS_DIR, exist_ok=True)
    api = HfApi()
    for repo_id, info in MODELS_TO_DOWNLOAD.items():
        print(f"\nProcessing repository: {repo_id}")
        local_dir_path = os.path.join(MODELS_DIR, info["local_dir"])
        try:
            repo_files = api.model_info(repo_id=repo_id).siblings
            file_sizes = {f.rfilename: f.size for f in repo_files}
        except Exception as e:
            print(f"  ⚠️  Could not fetch repository info. Sizes won't be displayed. Error: {e}")
            file_sizes = {}
        for filename in info["files"]:
            local_filepath = os.path.join(local_dir_path, filename)
            if os.path.exists(local_filepath):
                print(f"  ✅ File '{filename}' already exists. Skipping.")
                continue
            size = file_sizes.get(filename)
            print(f"  📥 Downloading '{filename}' ({format_size(size)})...")
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=local_dir_path,
                    local_dir_use_symlinks=False
                )
            except Exception as e:
                print(f"  ❌ FAILED to download '{filename}'. Error: {e}")
    print("\n--- Model Download Process Complete ---")

if __name__ == "__main__":
    download_all_models()