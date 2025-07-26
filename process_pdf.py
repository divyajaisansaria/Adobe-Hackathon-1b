import os
import json
import re
import argparse
from datetime import datetime
from collections import Counter
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer, CrossEncoder, models
from sklearn.metrics.pairwise import cosine_similarity
from llama_cpp import Llama


def read_json(path):
    """Reads and parses a JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_json(data, path):
    """Writes data to a JSON file, creating directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def get_timestamp():
    """Returns the current UTC time in ISO format."""
    return datetime.utcnow().isoformat()

def format_extracted_sections(ranked_chunks):
    """Formats ranked chunks for the 'extracted_sections' part of the output."""
    return [{"document": c["document"], "section_title": c["title"], "importance_rank": i + 1, "page_number": c["page"]} for i, c in enumerate(ranked_chunks)]

def format_subsection_analysis(ranked_chunks):
    """Formats ranked chunks for the 'subsection_analysis' part of the output."""
    return [{"document": c["document"], "refined_text": c["text"][:1500], "page_number": c["page"]} for c in ranked_chunks]

def is_junk_line(line_text):
    """Checks if a line is likely a header, footer, or other non-content text."""
    text = line_text.strip().lower()
    if re.search(r"^(page\s*\d+|version\s*[\d\.]+|\d+\s*of\s*\d+)", text) or \
       ("..." in text and len(text.split()) > 3) or \
       ("copyright" in text or "©" in text or "all rights reserved" in text) or \
       (text.isnumeric()):
        return True
    return False

def get_level_from_structure(text):
    """Determines heading level (H1-H4) from text patterns like '1.1' or 'A.2'."""
    text = text.strip()
    if re.match(r"^\d+\.\d+\.\d+\.\d+(\s|\.)", text) or re.match(r"^[a-z]\.[a-z]\.[a-z]\.[a-z](\s|\.)", text): return "H4"
    if re.match(r"^\d+\.\d+\.\d+(\s|\.)", text) or re.match(r"^[a-z]\.[a-z]\.[a-z](\s|\.)", text): return "H3"
    if re.match(r"^\d+\.\d+(\s|\.)", text) or re.match(r"^[A-Z]\.\d+(\s|\.)", text): return "H2"
    if re.match(r"^(chapter|section|part)\s+[IVXLC\d]+", text, re.IGNORECASE) or \
       re.match(r"^\d+\.\s", text) or re.match(r"^[A-Z]\.\s", text): return "H1"
    return None

def extract_lines_and_features(pdf_path):
    """Extracts all text lines from a PDF and computes features for each line."""
    with pdfplumber.open(pdf_path) as pdf:
        all_lines, all_words = [], []
        for page in pdf.pages:
            all_words.extend(page.extract_words(extra_attrs=["size", "fontname"]))
        if not all_words: return [], {}
        font_sizes = [w["size"] for w in all_words if w["size"] is not None]
        doc_stats = {"most_common_font_size": Counter(font_sizes).most_common(1)[0][0] if font_sizes else 12.0}
        for page in pdf.pages:
            page_words = sorted(page.extract_words(extra_attrs=["size", "fontname", "bottom"]), key=lambda w: (w["top"], w["x0"]))
            if not page_words: continue
            current_line, current_top = [], page_words[0]["top"]
            for word in page_words:
                if abs(word["top"] - current_top) > 2:
                    if current_line: all_lines.append(build_line_object(current_line, page))
                    current_line, current_top = [word], word["top"]
                else:
                    current_line.append(word)
            if current_line: all_lines.append(build_line_object(current_line, page))
        for i, line in enumerate(all_lines):
            prev_line_bottom = all_lines[i - 1]["bottom"] if i > 0 and all_lines[i - 1]["page"] == line["page"] else 0
            line["gap_before"] = line["top"] - prev_line_bottom
    return all_lines, doc_stats

def build_line_object(words, page):
    """Constructs a line object from a list of word objects."""
    text = " ".join(w["text"] for w in words)
    sizes = [w["size"] for w in words]
    names = [w["fontname"] for w in words]
    return {"text": text, "page": page.page_number - 1, "top": words[0]["top"], "bottom": max(w["bottom"] for w in words), "font_size": np.mean(sizes), "is_bold": any("bold" in name.lower() for name in names), "word_count": len(words)}

def score_headings(lines, doc_stats):
    """Scores lines based on features to identify potential headings."""
    scored_lines = []
    if not lines: return []
    body_font_size = doc_stats.get("most_common_font_size", 12.0)
    for line in lines:
        if is_junk_line(line["text"]): continue
        score = 0
        if line["font_size"] > body_font_size * 1.15: score += 20
        if line["is_bold"]: score += 15
        if line["gap_before"] > line["font_size"] * 1.5: score += 15
        if line["word_count"] <= 12: score += 10
        if line["word_count"] > 20: score -= 15
        if score > 25:
            line["score"] = score
            scored_lines.append(line)
    return scored_lines

def classify_and_build_outline(potential_headings, lines):
    """Classifies headings by level and style to build a document outline."""
    if not potential_headings: return [], lines[0]["text"] if lines else "No Title Found"
    title_candidates = sorted([h for h in potential_headings if h["page"] == 0 and h["top"] < 200], key=lambda x: x["top"])
    title_text, title_lines = "", []
    if title_candidates:
        primary_title_line = max(title_candidates, key=lambda x: x.get('score', 0), default=None)
        if primary_title_line:
            title_lines.append(primary_title_line)
            for cand in title_candidates:
                if cand not in title_lines and abs(cand["top"] - title_lines[-1]["bottom"]) < 25: title_lines.append(cand)
            title_lines.sort(key=lambda x: x["top"])
            title_text = " ".join(line["text"] for line in title_lines)
    title_texts = {line["text"] for line in title_lines}
    headings_to_classify = [h for h in potential_headings if h["text"] not in title_texts]
    outline, unclassified = [], []
    for h in headings_to_classify:
        level = get_level_from_structure(h["text"])
        if level: outline.append({"level": level, "text": h["text"].strip(), "page": h["page"]})
        else: unclassified.append(h)
    if unclassified:
        fallback_styles = sorted(list(set((h["font_size"], h["is_bold"]) for h in unclassified)), key=lambda x: x[0], reverse=True)
        level_map, h1_found = {}, any(o["level"] == "H1" for o in outline)
        if fallback_styles and not h1_found: level_map[fallback_styles[0]] = "H1"
        if len(fallback_styles) > 1: level_map[fallback_styles[1 if not h1_found else 0]] = "H2"
        for style in fallback_styles[2 if not h1_found else 1:]: level_map[style] = "H3"
        for h in unclassified:
            style = (h["font_size"], h["is_bold"])
            outline.append({"level": level_map.get(style, "H3"), "text": h["text"].strip(), "page": h["page"]})
    line_positions = {line["text"]: i for i, line in enumerate(lines)}
    outline.sort(key=lambda x: (x["page"], line_positions.get(x["text"], 0)))
    return outline, title_text.strip()

def extract_pdf_text_chunks(pdf_path):
    """Extracts structured text chunks from a PDF based on its outline."""
    lines, doc_stats = extract_lines_and_features(pdf_path)
    if not lines: return []
    potential_headings = score_headings(lines, doc_stats)
    outline, _ = classify_and_build_outline(potential_headings, lines)
    chunks, line_map = [], {(l["page"], l["text"]): i for i, l in enumerate(lines)}
    sorted_outline = sorted(outline, key=lambda h: (h["page"], line_map.get((h["page"], h["text"]), float('inf'))))

    for i, heading in enumerate(sorted_outline):
        start_key = (heading["page"], heading["text"])
        if start_key not in line_map: continue
        start_index = line_map[start_key]
        end_index = len(lines)
        if i + 1 < len(sorted_outline):
            next_heading = sorted_outline[i+1]
            end_key = (next_heading["page"], next_heading["text"])
            if end_key in line_map:
                end_index = line_map[end_key]
        
        content_lines = lines[start_index + 1 : end_index]
        # Filter content to ensure it's on the correct pages before the next heading
        filtered_content = []
        for l in content_lines:
            if i + 1 < len(sorted_outline):
                next_heading = sorted_outline[i+1]
                if l['page'] > next_heading['page'] or (l['page'] == next_heading['page'] and l['top'] >= lines[line_map[end_key]]['top']):
                    break
            if not is_junk_line(l['text']):
                filtered_content.append(l['text'])

        content_text = "\n".join(filtered_content)
        chunks.append({"title": heading["text"], "text": content_text.strip(), "page": heading["page"]})
    return chunks

class TinyLlamaQueryGenerator:
    """Uses TinyLlama to generate search queries based on a persona and task."""
    def __init__(self, model_dir):
        model_path = os.path.join(model_dir, "tinyllama-1.1b-chat-v1.0-gguf", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
        if not os.path.exists(model_path): raise FileNotFoundError(f"Llama model not found at {model_path}.")
        self.llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)

    def generate_queries(self, persona, task, num_queries=5):
        prompt = f"Persona: {persona}\nTask: {task}\nGenerate {num_queries} diverse search queries:"
        output = self.llm(prompt, max_tokens=150, stop=["\n\n"])
        raw_text = output["choices"][0]["text"]
        queries = [line.strip("-1234567890. ") for line in raw_text.split("\n") if len(line.strip()) > 10]
        if not queries: return [f"Query: overview of {task}", f"Query: key concepts in {task}"]
        return [f"Query: {q}" for q in queries[:num_queries]]

class MultiQueryRanker:
    """Ranks text chunks using a multi-query and cross-encoder approach."""
    def __init__(self, model_dir, use_tiny_llama=True):
        print("Initializing MultiQueryRanker...")
        bge_model_path = os.path.join(model_dir, "bge-small-en-v1.5")
        cross_encoder_path = os.path.join(model_dir, "cross-encoder-ms-marco")

        word_embedding_model = models.Transformer(bge_model_path)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        self.cross_encoder = CrossEncoder(cross_encoder_path)
        self.query_generator = TinyLlamaQueryGenerator(model_dir=model_dir) if use_tiny_llama else None
        print("Initialization complete.")
        
    def rank(self, persona, task, chunks, top_k=5, max_chunks_per_doc=2):
        queries = self.query_generator.generate_queries(persona, task) if self.query_generator else [f"Query: overview of {task}"]
        print(f"Generated {len(queries)} queries for ranking.")
        
        if not chunks:
            return []
            
        query_embeddings = self.model.encode(queries)
        chunk_texts = [c["text"] for c in chunks]
        chunk_embeddings = self.model.encode(chunk_texts)
        sim_scores = cosine_similarity(query_embeddings, chunk_embeddings)
        weights = [0.4] + [0.6 / (len(queries) - 1)] * (len(queries) - 1) if len(queries) > 1 else [1.0]
        
        for i, chunk in enumerate(chunks):
            chunk["similarity"] = float(np.average(sim_scores[:, i], weights=weights))
            
        ranked = sorted(chunks, key=lambda x: x["similarity"], reverse=True)
        doc_counter, filtered = {}, []
        
        for chunk in ranked:
            if doc_counter.get(chunk["document"], 0) < max_chunks_per_doc:
                filtered.append(chunk)
                doc_counter[chunk["document"]] = doc_counter.get(chunk["document"], 0) + 1
            if len(filtered) >= top_k * 2: break
                
        if not filtered: return []
        
        pairs = [(task, chunk["text"]) for chunk in filtered]
        cross_scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        
        for i, chunk in enumerate(filtered):
            chunk["cross_score"] = float(cross_scores[i])
            
        return sorted(filtered, key=lambda x: x["cross_score"], reverse=True)[:top_k]

def main(input_dir, output_dir, model_dir):
    """Main function to process PDFs from a single input directory."""
    print("-" * 80)
    print(f"Starting processing for input directory: {input_dir}")

    # --- 1. Define and Validate Paths ---
    input_json_path = os.path.join(input_dir, "input.json")
    pdfs_dir = os.path.join(input_dir, "PDFs")
    output_json_path = os.path.join(output_dir, "output.json")

    if not os.path.exists(input_json_path):
        print(f"❌ Error: 'input.json' not found in '{input_dir}'.")
        return
    if not os.path.isdir(pdfs_dir):
        print(f"❌ Error: 'PDFs' directory not found in '{input_dir}'.")
        return

    # --- 2. Load Input Data and Initialize ---
    os.makedirs(output_dir, exist_ok=True)
    try:
        ranker = MultiQueryRanker(model_dir=model_dir)
        input_data = read_json(input_json_path)
    except Exception as e:
        print(f"❌ Error during initialization or reading input JSON: {e}")
        return

    persona = input_data.get("persona", {}).get("role", "user")
    task = input_data.get("job_to_be_done", {}).get("task", "summarize")
    pdf_files = [f for f in os.listdir(pdfs_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("-> Skipping: No PDF files found in the 'PDFs' directory.")
        return

    print(f"Found {len(pdf_files)} PDF(s) to process.")

    # --- 3. Extract Text from all PDFs ---
    all_chunks = []
    for file_name in pdf_files:
        print(f"  -> Extracting text from: {file_name}")
        pdf_path = os.path.join(pdfs_dir, file_name)
        try:
            chunks = extract_pdf_text_chunks(pdf_path)
            for chunk in chunks:
                chunk["document"] = file_name
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"     -> Warning: Could not process {file_name}. Error: {e}")

    if not all_chunks:
        print("❌ Error: No text chunks could be extracted from any of the provided PDFs.")
        return

    # --- 4. Rank Chunks and Generate Output ---
    print("Ranking extracted chunks...")
    ranked_chunks = ranker.rank(persona, task, all_chunks)

    result = {
        "metadata": {
            "input_documents": pdf_files,
            "persona": persona,
            "job_to_be_done": input_data.get("job_to_be_done", {})
        },
        "extracted_sections": format_extracted_sections(ranked_chunks),
        "subsection_analysis": format_subsection_analysis(ranked_chunks)
    }
    
    # --- 5. Write Final Output ---
    write_json(result, output_json_path)
    print("-" * 80)
    print(f"✅ Success! Result written to: {output_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a directory of documents based on an input.json file.")
    parser.add_argument("input_dir", help="The directory containing 'input.json' and the 'PDFs' folder.")
    parser.add_argument("output_dir", help="The directory where 'output.json' will be saved.")
    parser.add_argument("model_dir", help="The directory containing the model files.")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.model_dir)