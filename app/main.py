#main.py
import os
from app.pdf_reader import extract_pdf_text_chunks
from app.multiquery_ranker import MultiQueryRanker
from app.utils import read_json, write_json, get_timestamp, format_extracted_sections, format_subsection_analysis

def run_pipeline(input_dir, output_dir):
    persona_path = os.path.join(input_dir, "persona.json")
    persona_data = read_json(persona_path)
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]

    # Flatten persona if it's a dict with 'role'
    persona = persona_data["persona"]
    if isinstance(persona, dict) and "role" in persona:
        persona = persona["role"]

    task = persona_data["job_to_be_done"]["task"]

    all_chunks = []
    for file_name in pdf_files:
        pdf_path = os.path.join(input_dir, file_name)
        chunks = extract_pdf_text_chunks(pdf_path)
        for chunk in chunks:
            chunk["document"] = file_name
        all_chunks.extend(chunks)

    ranker = MultiQueryRanker()
    ranked_chunks = ranker.rank(persona, task, all_chunks)

    result = {
        "metadata": {
            "input_documents": pdf_files,
            "persona": persona,  # flattened string
            "job_to_be_done": persona_data["job_to_be_done"],
            "processing_timestamp": get_timestamp()
        },
        "extracted_sections": format_extracted_sections(ranked_chunks),
        "subsection_analysis": format_subsection_analysis(ranked_chunks)
    }

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "result.json")
    write_json(result, output_path)
    print(f"✅ Result written to: {output_path}")

if __name__ == "__main__":
    run_pipeline("input/Collection1", "output/Collection1")
