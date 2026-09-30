"""
FilingLens — Stage 4a: Search the local embedding index
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import sys


INDEX_DIR = "index"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5


def load_index():
    embeddings = np.load(f"{INDEX_DIR}/embeddings.npy")
    with open(f"{INDEX_DIR}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return embeddings, chunks


def cosine_similarity(query_vec, matrix):
    query_norm = query_vec / np.linalg.norm(query_vec)
    matrix_norms = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix_norms @ query_norm


def search(query, model, embeddings, chunks, top_k=TOP_K):
    query_vec = model.encode(query)
    scores = cosine_similarity(query_vec, embeddings)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "text": chunks[idx]["text"],
            "section": chunks[idx]["section"],
            "filing": chunks[idx]["filing"],
            "score": round(float(scores[idx]), 3),
        })
    return results


if __name__ == "__main__":
    print(f"Loading embedding model ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)

    print("Loading index...")
    embeddings, chunks = load_index()
    print(f"Index loaded: {len(chunks)} chunks.\n")

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter a question: ")

    results = search(query, model, embeddings, chunks)

    print(f"\nTop {len(results)} results for: \"{query}\"\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['filing']} | {r['section']} | score={r['score']}]")
        print(f"   {r['text'][:250]}")
        print()
