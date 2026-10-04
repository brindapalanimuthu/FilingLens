"""
FilingLens — Stage 5: Combined text + table index (multi-company)

Reads section labels from classified_deberta/ when a file exists there,
otherwise falls back to classified/. Every entry gets `company` (ticker)
and `fiscal_year` fields.
"""

from sentence_transformers import SentenceTransformer
import json
import os
import glob
import re
import numpy as np
import pickle
from table_chunker import table_to_row_chunks


# classified/ first, then classified_deberta/ overrides it per filing
INPUT_DIRS = ["classified", "classified_deberta"]
OUTPUT_DIR = "full_index"
MODEL_NAME = "all-MiniLM-L6-v2"

TICKER_NAMES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
}


def company_info(filing_name):
    """'MSFT_10-K_2025-07-30' -> ('MSFT', 'Microsoft Corporation', 2025).
    Fiscal year = filing year, which holds for both Apple (Sept FY, filed
    Oct/Nov) and Microsoft (June FY, filed July)."""
    ticker = filing_name.split("_")[0].upper()
    m = re.search(r"(\d{4})-\d{2}-\d{2}", filing_name)
    year = int(m.group(1)) if m else None
    return ticker, TICKER_NAMES.get(ticker, ticker), year


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

    files = {}
    for d in INPUT_DIRS:
        for p in glob.glob(os.path.join(d, "*.json")):
            files[os.path.basename(p)] = p
    print(f"Found {len(files)} classified filing(s) to index:")
    for name, p in sorted(files.items()):
        print(f"  {name}  <- {os.path.dirname(p)}/")

    all_entries = []
    for fname, classified_path in sorted(files.items()):
        with open(classified_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        filing_name = os.path.splitext(fname)[0]
        ticker, company_name, fiscal_year = company_info(filing_name)

        # text chunks
        for chunk in data["text"]:
            all_entries.append({
                "type": "text",
                "embed_text": chunk["text"],
                "display_text": chunk["text"],
                "section": chunk["section"],
                "filing": filing_name,
                "company": ticker,
                "fiscal_year": fiscal_year,
            })

        # table rows (one entry per row)
        for table in data["tables"]:
            if is_boilerplate_table(table["rows"]):
                continue
            for row_chunk in table_to_row_chunks(table, filing_name, company_name):
                all_entries.append({
                    "type": "table",
                    "embed_text": row_chunk["text"],
                    "display_text": row_chunk["text"],
                    "section": f"Table {table['table_id']}",
                    "filing": filing_name,
                    "company": ticker,
                    "fiscal_year": fiscal_year,
                })

    print(f"\nEmbedding {len(all_entries)} total entries "
          f"(text + tables) across all filings...")
    texts = [e["embed_text"] for e in all_entries]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    np.save(os.path.join(OUTPUT_DIR, "embeddings.npy"), embeddings)
    with open(os.path.join(OUTPUT_DIR, "entries.pkl"), "wb") as f:
        pickle.dump(all_entries, f)

    print(f"\nSaved {len(all_entries)} entries to {OUTPUT_DIR}/")
    for t in sorted({e['company'] for e in all_entries}):
        rows = [e for e in all_entries if e["company"] == t]
        n_text = sum(1 for e in rows if e["type"] == "text")
        n_table = sum(1 for e in rows if e["type"] == "table")
        print(f"  {t}: {len(rows)} entries ({n_text} text, {n_table} table rows)")