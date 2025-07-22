#heading_extractor.py
import os
import json
import re
import pdfplumber
import numpy as np
from pathlib import Path
from collections import Counter

def is_junk_line(line_text):
    text = line_text.strip().lower()
    if re.search(r"^(page\s*\d+|version\s*[\d\.]+|\d+\s*of\s*\d+)", text) or \
       ("..." in text and len(text.split()) > 3) or \
       ("copyright" in text or "©" in text or "all rights reserved" in text) or \
       (text.isnumeric()):
        return True
    return False

def get_level_from_structure(text):
    text = text.strip()
    if re.match(r"^\d+\.\d+\.\d+\.\d+(\s|\.)", text) or re.match(r"^[a-z]\.[a-z]\.[a-z]\.[a-z](\s|\.)", text):
        return "H4"
    if re.match(r"^\d+\.\d+\.\d+(\s|\.)", text) or re.match(r"^[a-z]\.[a-z]\.[a-z](\s|\.)", text):
        return "H3"
    if re.match(r"^\d+\.\d+(\s|\.)", text) or re.match(r"^[A-Z]\.\d+(\s|\.)", text):
        return "H2"
    if re.match(r"^(chapter|section|part)\s+[IVXLC\d]+", text, re.IGNORECASE) or \
       re.match(r"^\d+\.\s", text) or re.match(r"^[A-Z]\.\s", text):
        return "H1"
    return None

def extract_lines_and_features(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        all_lines, all_words = [], []
        for page in pdf.pages:
            all_words.extend(page.extract_words(extra_attrs=["size", "fontname"]))
        if not all_words:
            return [], {}
        font_sizes = [w["size"] for w in all_words]
        doc_stats = { "most_common_font_size": Counter(font_sizes).most_common(1)[0][0] }

        for page in pdf.pages:
            page_words = sorted(page.extract_words(extra_attrs=["size", "fontname", "bottom"]),
                                key=lambda w: (w["top"], w["x0"]))
            if not page_words:
                continue
            current_line, current_top = [], page_words[0]["top"]
            for word in page_words:
                if abs(word["top"] - current_top) > 2:
                    if current_line:
                        all_lines.append(build_line_object(current_line, page))
                    current_line = [word]
                    current_top = word["top"]
                else:
                    current_line.append(word)
            if current_line:
                all_lines.append(build_line_object(current_line, page))

        for i, line in enumerate(all_lines):
            prev_line_bottom = all_lines[i-1]["bottom"] if i > 0 and all_lines[i-1]["page"] == line["page"] else 0
            line["gap_before"] = line["top"] - prev_line_bottom

    return all_lines, doc_stats

def build_line_object(words, page):
    text = " ".join(w["text"] for w in words)
    sizes = [w["size"] for w in words]
    names = [w["fontname"] for w in words]
    return {
        "text": text,
        "page": page.page_number - 1,
        "top": words[0]["top"],
        "bottom": max(w["bottom"] for w in words),
        "font_size": np.mean(sizes),
        "is_bold": any("bold" in name.lower() for name in names),
        "word_count": len(words)
    }

def score_headings(lines, doc_stats):
    scored_lines = []
    if not lines:
        return []
    body_font_size = doc_stats["most_common_font_size"]
    for line in lines:
        if is_junk_line(line["text"]):
            continue
        score = 0
        if line["font_size"] > body_font_size * 1.15:
            score += 20
        if line["is_bold"]:
            score += 15
        if line["gap_before"] > line["font_size"] * 1.5:
            score += 15
        if line["word_count"] <= 12:
            score += 10
        if line["word_count"] > 20:
            score -= 15
        if score > 25:
            line["score"] = score
            scored_lines.append(line)
    return scored_lines

def classify_and_build_outline(potential_headings, lines):
    if not potential_headings:
        title_text = lines[0]["text"] if lines else "No Title Found"
        return [], title_text

    title_candidates = sorted(
        [h for h in potential_headings if h["page"] == 0 and h["top"] < 200],
        key=lambda x: x["top"]
    )
    title_text = ""
    title_lines = []
    if title_candidates:
        primary_title_line = max(title_candidates, key=lambda x: x.get('score', 0), default=None)
        if primary_title_line:
            title_lines.append(primary_title_line)
            for cand in title_candidates:
                if cand not in title_lines and abs(cand["top"] - title_lines[-1]["bottom"]) < 25:
                    title_lines.append(cand)
            title_lines.sort(key=lambda x: x["top"])
            title_text = " ".join(line["text"] for line in title_lines)

    title_texts = {line["text"] for line in title_lines}
    headings_to_classify = [h for h in potential_headings if h["text"] not in title_texts]

    outline = []
    unclassified_by_structure = []

    for h in headings_to_classify:
        level = get_level_from_structure(h["text"])
        if level:
            outline.append({"level": level, "text": h["text"].strip(), "page": h["page"]})
        else:
            unclassified_by_structure.append(h)

    if unclassified_by_structure:
        fallback_styles = sorted(
            list(set((h["font_size"], h["is_bold"]) for h in unclassified_by_structure)),
            key=lambda x: x[0], reverse=True
        )
        level_map = {}
        h1_found = any(o["level"] == "H1" for o in outline)
        if fallback_styles and not h1_found:
            level_map[fallback_styles[0]] = "H1"
        if len(fallback_styles) > 1:
            level_map[fallback_styles[1 if not h1_found else 0]] = "H2"
        for style in fallback_styles[2 if not h1_found else 1:]:
            level_map[style] = "H3"
        for h in unclassified_by_structure:
            style = (h["font_size"], h["is_bold"])
            level = level_map.get(style, "H3")
            outline.append({"level": level, "text": h["text"].strip(), "page": h["page"]})

    line_positions = {line["text"]: i for i, line in enumerate(lines)}
    outline.sort(key=lambda x: (x["page"], line_positions.get(x["text"], 0)))
    return outline, title_text.strip()
