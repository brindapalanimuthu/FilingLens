"""
FilingLens — Stage 5: Generate an answer over text AND tables, with
numeric verification.
"""

from google import genai
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import os
import re
import sys


INDEX_DIR = "full_index"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.5-flash"
TOP_K = 5

NUMBER_PATTERN = re.compile(r"\(?-?\$?\d[\d,]*(?:\.\d+)?\)?%?")


def load_index():
    embeddings = np.load(f"{INDEX_DIR}/embeddings.npy")
    with open(f"{INDEX_DIR}/entries.pkl", "rb") as f:
        entries = pickle.load(f)
    return embeddings, entries


def cosine_similarity(query_vec, matrix):
    query_norm = query_vec / np.linalg.norm(query_vec)
    matrix_norms = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix_norms @ query_norm


def search(query, embed_model, embeddings, entries, top_k=TOP_K):
    query_vec = embed_model.encode(query)
    scores = cosine_similarity(query_vec, embeddings)

    text_indices = [i for i, e in enumerate(entries) if e["type"] == "text"]
    table_indices = [i for i, e in enumerate(entries) if e["type"] == "table"]

    n_table = min(2, len(table_indices))
    n_text = top_k - n_table

    top_text = sorted(text_indices, key=lambda i: -scores[i])[:n_text]
    top_table = sorted(table_indices, key=lambda i: -scores[i])[:n_table]

    combined = sorted(top_text + top_table, key=lambda i: -scores[i])
    return [entries[i] for i in combined]


def build_prompt(query, retrieved):
    context_block = "\n\n".join(
        f"[Source {i+1}: {e['filing']}, {e['section']}, type: {e['type']}]\n{e['display_text']}"
        for i, e in enumerate(retrieved)
    )
    return f"""You are a financial analyst assistant. Answer the question using ONLY
the sources below. Some sources are tables (rows of numbers) and some are
prose. Cite sources by their number (e.g. "[Source 2]"). When citing a
number from a table, quote it exactly as it appears — do not round or
recalculate. If the sources don't contain enough information to answer
confidently, say so explicitly rather than guessing.

SOURCES:
{context_block}

QUESTION: {query}

ANSWER:"""


def verify_numbers(answer_text, retrieved):
    def normalize(s):
        return re.sub(r"[,$%()]", "", s)

    source_text = " ".join(e["display_text"] for e in retrieved)
    normalized_source = normalize(source_text)

    numbers_found = NUMBER_PATTERN.findall(answer_text)
    results = []
    seen = set()
    for num in numbers_found:
        if num in seen:
            continue
        seen.add(num)
        normalized_num = normalize(num)
        if len(normalized_num) < 2:
            continue
        verified = normalized_num in normalized_source
        results.append({"number": num, "verified": verified})
    return results


if __name__ == "__main__":
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY as an environment variable first.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    print(f"Loading embedding model ({EMBED_MODEL_NAME})...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Loading combined text+table index...")
    embeddings, entries = load_index()
    print(f"Index loaded: {len(entries)} entries.\n")

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter a question: ")

    retrieved = search(query, embed_model, embeddings, entries)
    prompt = build_prompt(query, retrieved)

    print("\nRetrieved sources:")
    for i, e in enumerate(retrieved, 1):
        print(f"  [{i}] {e['filing']} | {e['section']} | type={e['type']}")

    print("\nAsking Gemini...\n")
    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=prompt,
    )

    print("=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(response.text)

    verification = verify_numbers(response.text, retrieved)
    print("\n" + "=" * 60)
    print("VERIFICATION:")
    print("=" * 60)
    if not verification:
        print("No numeric claims detected to verify.")
    else:
        for v in verification:
            status = "found in sources" if v["verified"] else "NOT FOUND in sources"
            marker = "OK " if v["verified"] else "!! "
            print(f"  {marker}{v['number']:>15}  {status}")
        unverified = [v for v in verification if not v["verified"]]
        if unverified:
            print(f"\n{len(unverified)} number(s) could not be verified against "
                  f"the retrieved sources — treat this answer with caution.")
        else:
            print(f"\nAll {len(verification)} number(s) in the answer were "
                  f"confirmed present in the retrieved sources.")
