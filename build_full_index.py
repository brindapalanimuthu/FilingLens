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
from table_chunker import table_to_row_chunks


INPUT_DIR = "classified"
OUTPUT_DIR = "full_index"
MODEL_NAME = "all-MiniLM-L6-v2"


def is_boilerplate_table(rows):
    joined = " ".join(cell for row in rows for cell in row)
    boilerplate_markers = [
        "Incorporated by Reference", "Exhibit Number", "/s/",
        "PCAOB Firm ID", "Power of Attorney", "Rule 10b5-1",
    ]
    return any(marker in joined for marker in boilerplate_markers)


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

        # text chunks
        for chunk in data["text"]:
            all_entries.append({
                "type": "text",
                "embed_text": chunk["text"],
                "display_text": chunk["text"],
                "section": chunk["section"],
                "filing": filing_name,
            })

        # table rows (one entry per row)
        for table in data["tables"]:
            if is_boilerplate_table(table["rows"]):
                continue
            for row_chunk in table_to_row_chunks(table, filing_name):
                all_entries.append({
                    "type": "table",
                    "embed_text": row_chunk["text"],
                    "display_text": row_chunk["text"],
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