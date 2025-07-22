# utils.py
import json
import os
from datetime import datetime

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def get_timestamp():
    return datetime.utcnow().isoformat()

def format_extracted_sections(ranked_chunks):
    return [{
        "document": chunk["document"],
        "section_title": chunk["title"],
        "importance_rank": i + 1,
        "page_number": chunk["page"]
    } for i, chunk in enumerate(ranked_chunks)]

def format_subsection_analysis(ranked_chunks):
    return [{
        "document": chunk["document"],
        "refined_text": chunk["text"][:1500],
        "page_number": chunk["page"]
    } for chunk in ranked_chunks]
