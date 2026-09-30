"""
FilingLens — Stage 6: LangGraph routing (hybrid table retrieval)
"""

from langgraph.graph import StateGraph, END
from google import genai
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import os
import re
import sys
from typing import TypedDict, List, Dict

from hybrid_search import hybrid_table_search


INDEX_DIR = "full_index"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"
TOP_K = 5

NUMBER_PATTERN = re.compile(r"\(?-?\$?\d[\d,]*(?:\.\d+)?\)?%?")

NUMERIC_KEYWORDS = [
    "how much", "total", "percentage", "percent", "%", "margin",
    "revenue", "net sales", "net income", "eps", "earnings per share",
    "cash", "debt", "compare", "increase", "decrease", "growth",
    "rate", "ratio", "cost", "expense", "profit", "gross margin",
    "spend", "assets", "equity", "shareholders",
]

EXPLANATORY_KEYWORDS = [
    "why", "what caused", "what led", "reason", "explain", "describe",
    "how did", "what factors", "what risks",
]


class GraphState(TypedDict):
    query: str
    route: str
    retrieved: List[Dict]
    answer: str
    verification: List[Dict]


def load_index():
    embeddings = np.load(f"{INDEX_DIR}/embeddings.npy")
    with open(f"{INDEX_DIR}/entries.pkl", "rb") as f:
        entries = pickle.load(f)
    return embeddings, entries


def cosine_similarity(query_vec, matrix):
    query_norm = query_vec / np.linalg.norm(query_vec)
    matrix_norms = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix_norms @ query_norm


def normalize(s):
    return re.sub(r"[,$%()]", "", s)


def call_gemini_with_retry(client, prompt, max_attempts=5):
    import time
    from google.genai.errors import ClientError

    for attempt in range(1, max_attempts + 1):
        try:
            return client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=prompt,
            )
        except ClientError as e:
            msg = str(e)
            # daily quota can't be fixed by waiting a few seconds
            if "PerDay" in msg:
                raise
            if "RESOURCE_EXHAUSTED" in msg and attempt < max_attempts:
                wait = 35
                print(f"  Rate limited, waiting {wait}s before retry "
                      f"({attempt}/{max_attempts})...")
                time.sleep(wait)
            else:
                raise


def build_graph(embed_model, embeddings, entries, gemini_client):

    def classify_query(state):
        query_lower = state["query"].lower()
        is_explanatory = any(kw in query_lower for kw in EXPLANATORY_KEYWORDS)
        if is_explanatory:
            return {**state, "route": "text_focused"}
        is_numeric = any(kw in query_lower for kw in NUMERIC_KEYWORDS)
        return {**state, "route": "table_focused" if is_numeric else "text_focused"}

    def retrieve(state, n_table, n_text):
        query_vec = embed_model.encode(state["query"])
        scores = cosine_similarity(query_vec, embeddings)

        text_indices = [i for i, e in enumerate(entries) if e["type"] == "text"]
        table_indices = [i for i, e in enumerate(entries) if e["type"] == "table"]

        top_text = sorted(text_indices, key=lambda i: -scores[i])[:n_text]
        top_table = sorted(table_indices, key=lambda i: -scores[i])[:min(n_table, len(table_indices))]

        combined = sorted(top_text + top_table, key=lambda i: -scores[i])
        return {**state, "retrieved": [entries[i] for i in combined]}

    def retrieve_table_focused(state):
        query = state["query"]

        # hybrid BM25 + embedding search over table rows only
        tables = hybrid_table_search(query, k=12, model=embed_model)

        # if the question names a year, put rows from that year's filing first
        # (filings are named by filing date, e.g. AAPL_10-K_2024-11-01)
        years = re.findall(r"\b(20\d{2})\b", query)
        if years:
            tables = sorted(
                tables,
                key=lambda e: 0 if f"_{years[0]}-" in e["filing"] else 1,
            )
        tables = tables[:6]

        # a couple of prose chunks for context (n_table=0 -> text only)
        text_state = retrieve(state, n_table=0, n_text=2)
        return {**state, "retrieved": tables + text_state["retrieved"]}

    def retrieve_text_focused(state):
        return retrieve(state, n_table=1, n_text=4)

    def generate(state):
        context_block = "\n\n".join(
            f"[Source {i+1}: {e['filing']}, {e['section']}, type: {e['type']}]\n{e['display_text']}"
            for i, e in enumerate(state["retrieved"])
        )
        prompt = f"""You are a financial analyst assistant. Answer the question using ONLY
the sources below. Some sources are tables (rows of numbers) and some are
prose. Cite sources by their number (e.g. "[Source 2]"). When citing a
number from a table, quote it exactly as it appears — do not round or
recalculate. When the question names a fiscal year, prefer figures from the
filing for that same year (e.g. the 2024 10-K for 2024 figures). If the
sources don't contain enough information to answer confidently, say so
explicitly rather than guessing.

SOURCES:
{context_block}

QUESTION: {state['query']}

ANSWER:"""
        response = call_gemini_with_retry(gemini_client, prompt)
        return {**state, "answer": response.text}

    def verify(state):
        source_text = " ".join(e["display_text"] for e in state["retrieved"])
        normalized_source = normalize(source_text)

        numbers_found = NUMBER_PATTERN.findall(state["answer"])
        results = []
        seen = set()
        for num in numbers_found:
            if num in seen:
                continue
            seen.add(num)
            normalized_num = normalize(num)
            if len(normalized_num) < 2:
                continue
            results.append({
                "number": num,
                "verified": normalized_num in normalized_source,
            })
        return {**state, "verification": results}

    graph = StateGraph(GraphState)
    graph.add_node("classify_query", classify_query)
    graph.add_node("retrieve_table_focused", retrieve_table_focused)
    graph.add_node("retrieve_text_focused", retrieve_text_focused)
    graph.add_node("generate", generate)
    graph.add_node("verify", verify)

    graph.set_entry_point("classify_query")
    graph.add_conditional_edges(
        "classify_query",
        lambda state: state["route"],
        {
            "table_focused": "retrieve_table_focused",
            "text_focused": "retrieve_text_focused",
        },
    )
    graph.add_edge("retrieve_table_focused", "generate")
    graph.add_edge("retrieve_text_focused", "generate")
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", END)

    return graph.compile()


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

    app = build_graph(embed_model, embeddings, entries, client)

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter a question: ")

    result = app.invoke({"query": query, "route": "", "retrieved": [],
                          "answer": "", "verification": []})

    print(f"\nRoute chosen: {result['route']}")
    print("\nRetrieved sources:")
    for i, e in enumerate(result["retrieved"], 1):
        print(f"  [{i}] {e['filing']} | {e['section']} | type={e['type']}")

    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(result["answer"])

    print("\n" + "=" * 60)
    print("VERIFICATION:")
    print("=" * 60)
    verification = result["verification"]
    if not verification:
        print("No numeric claims detected to verify.")
    else:
        for v in verification:
            status = "found in sources" if v["verified"] else "NOT FOUND in sources"
            marker = "OK " if v["verified"] else "!! "
            print(f"  {marker}{v['number']:>15}  {status}")
        unverified = [v for v in verification if not v["verified"]]
        if unverified:
            print(f"\n{len(unverified)} number(s) could not be verified.")
        else:
            print(f"\nAll {len(verification)} number(s) confirmed present in sources.")