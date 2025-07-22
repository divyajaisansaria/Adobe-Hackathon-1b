# pdf_reader.py
from app.heading_extractor import extract_lines_and_features, score_headings, classify_and_build_outline

def extract_pdf_text_chunks(pdf_path):
    lines, doc_stats = extract_lines_and_features(pdf_path)
    if not lines:
        return []
    
    potential_headings = score_headings(lines, doc_stats)
    outline, _ = classify_and_build_outline(potential_headings, lines)

    chunks = []
    heading_lines = sorted(outline, key=lambda h: (h["page"], h["text"]))
    for i, heading in enumerate(heading_lines):
        start_index = lines.index(next(l for l in lines if l["text"] == heading["text"] and l["page"] == heading["page"]))
        end_index = (
            lines.index(next(l for l in lines if l["text"] == heading_lines[i + 1]["text"] and l["page"] == heading_lines[i + 1]["page"]))
            if i + 1 < len(heading_lines)
            else len(lines)
        )
        content_lines = lines[start_index + 1:end_index]
        content_text = "\n".join(l["text"] for l in content_lines if not l["text"].strip().lower().startswith("page"))
        chunks.append({
            "title": heading["text"],
            "text": content_text.strip(),
            "page": heading["page"]
        })
    return chunks
