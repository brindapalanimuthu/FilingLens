"""
FilingLens — Stage 4b: Generate an answer (RAG)
"""

from google import genai
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import os
import sys


INDEX_DIR = "index"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.5-flash"
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


def search(query, embed_model, embeddings, chunks, top_k=TOP_K):
    query_vec = embed_model.encode(query)
    scores = cosine_similarity(query_vec, embeddings)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices]


def build_prompt(query, retrieved_chunks):
    context_block = "\n\n".join(
        f"[Source {i+1}: {c['filing']}, section: {c['section']}]\n{c['text']}"
        for i, c in enumerate(retrieved_chunks)
    )
    return f"""You are a financial analyst assistant. Answer the question using ONLY
the sources below. Cite sources by their number (e.g. "[Source 2]").
If the sources don't contain enough information to answer confidently,
say so explicitly rather than guessing.

SOURCES:
{context_block}

QUESTION: {query}

ANSWER:"""


if __name__ == "__main__":
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY as an environment variable first.")
        print('  export GEMINI_API_KEY="your-key-here"')
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    print(f"Loading embedding model ({EMBED_MODEL_NAME})...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Loading index...")
    embeddings, chunks = load_index()
    print(f"Index loaded: {len(chunks)} chunks.\n")

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter a question: ")

    retrieved = search(query, embed_model, embeddings, chunks)
    prompt = build_prompt(query, retrieved)

    print("\nRetrieved sources:")
    for i, c in enumerate(retrieved, 1):
        print(f"  [{i}] {c['filing']} | {c['section']}")

    print("\nAsking Gemini...\n")
    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=prompt,
    )

    print("=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(response.text)
