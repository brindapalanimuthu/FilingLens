"""
FilingLens — Stage 4a: Build a local embedding index
"""

from sentence_transformers import SentenceTransformer
import json
import os
import glob
import numpy as np
import pickle


INPUT_DIR = "classified"
OUTPUT_DIR = "index"

MODEL_NAME = "all-MiniLM-L6-v2"


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading embedding model ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded.\n")

    classified_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    print(f"Found {len(classified_files)} classified filing(s) to index")

    all_chunks = []
    for classified_path in classified_files:
        with open(classified_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        filing_name = os.path.splitext(os.path.basename(classified_path))[0]
        for chunk in data["text"]:
            all_chunks.append({
                "text": chunk["text"],
                "section": chunk["section"],
                "filing": filing_name,
            })

    print(f"\nEmbedding {len(all_chunks)} total text chunks across all filings...")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    np.save(os.path.join(OUTPUT_DIR, "embeddings.npy"), embeddings)
    with open(os.path.join(OUTPUT_DIR, "chunks.pkl"), "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"\nSaved {len(all_chunks)} embeddings to {OUTPUT_DIR}/embeddings.npy")
    print(f"Saved chunk metadata to {OUTPUT_DIR}/chunks.pkl")
