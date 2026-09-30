"""
FilingLens — Stage 5: Combined text + table index
"""

from sentence_transformers import SentenceTransformer
import json
import os
import glob
import re
import numpy as np
import pickle


INPUT_DIR = "classified"
OUTPUT_DIR = "full_index"
MODEL_NAME = "all-MiniLM-L6-v2"


def clean_row(row):
    cleaned = [cell for cell in row if cell.strip()]
    merged = []
    i = 0
    while i < len(cleaned):
        if cleaned[i] == "$" and i + 1 < len(cleaned):
            merged.append(f"${cleaned[i + 1]}")
            i += 2
        else:
            merged.append(cleaned[i])
            i += 1
    return merged


def is_boilerplate_table(rows):
    joined = " ".join(cell for row in rows for cell in row)
    boilerplate_markers = [
        "Incorporated by Reference", "Exhibit Number", "/s/",
        "PCAOB Firm ID", "Power of Attorney", "Rule 10b5-1",
    ]
    return any(marker in joined for marker in boilerplate_markers)


def table_to_markdown(rows):
    cleaned_rows = [clean_row(r) for r in rows if clean_row(r)]
    return "\n".join(" | ".join(row) for row in cleaned_rows)


def is_numeric_like(cell):
    return bool(re.fullmatch(r"[\$\(\)\-–0-9,.% ]+", cell))


def table_to_embed_text(rows):
    cleaned_rows = [clean_row(r) for r in rows if clean_row(r)]
    if not cleaned_rows:
        return ""

    header = cleaned_rows[0]
    body_start = 1
    header_has_digit = any(re.search(r"\d", cell) for cell in header)
    if len(header) <= 2 and not header_has_digit and len(cleaned_rows) > 1:
        header = header + cleaned_rows[1]
        body_start = 2

    header_text = " ".join(header)
    labels = []
    for row in cleaned_rows[body_start:]:
        label = next((cell for cell in row if not is_numeric_like(cell)), None)
        if label:
            labels.append(label)

    return f"Columns: {header_text}. Rows: {', '.join(labels)}"


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading embedding model ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded.\n")

    classified_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    print(f"Found {len(classified_files)} classified filing(s) to index")

    all_entries = []
    for classified_path in classified_files:
        with open(classified_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        filing_name = os.path.splitext(os.path.basename(classified_path))[0]

        for chunk in data["text"]:
            all_entries.append({
                "type": "text",
                "embed_text": chunk["text"],
                "display_text": chunk["text"],
                "section": chunk["section"],
                "filing": filing_name,
            })

        for table in data["tables"]:
            rows = table["rows"]
            if is_boilerplate_table(rows):
                continue
            markdown = table_to_markdown(rows)
            if not markdown.strip():
                continue
            all_entries.append({
                "type": "table",
                "embed_text": table_to_embed_text(rows),
                "display_text": markdown,
                "section": f"Table {table['table_id']}",
                "filing": filing_name,
            })

    print(f"\nEmbedding {len(all_entries)} total entries "
          f"(text + tables) across all filings...")
    texts = [e["embed_text"] for e in all_entries]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    np.save(os.path.join(OUTPUT_DIR, "embeddings.npy"), embeddings)
    with open(os.path.join(OUTPUT_DIR, "entries.pkl"), "wb") as f:
        pickle.dump(all_entries, f)

    n_text = sum(1 for e in all_entries if e["type"] == "text")
    n_table = sum(1 for e in all_entries if e["type"] == "table")
    print(f"\nSaved {len(all_entries)} entries ({n_text} text, {n_table} tables) "
          f"to {OUTPUT_DIR}/")
