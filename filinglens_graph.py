"""
FilingLens — Stage 6: LangGraph routing (hybrid table retrieval, multi-company)
Gemini calls fall back between models when one hits its daily quota.
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

from hybrid_search import hybrid_table_search, detect_companies


INDEX_DIR = "full_index"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-3.5-flash-lite"]  # tried in this order
TOP_K = 5

_current_model_idx = 0  # persists across calls, so a dead model isn't retried

NUMBER_PATTERN = re.compile(r"\(?-?\$?\d[\d,]*(?:\.\d+)?\)?%?")

NUMERIC_KEYWORDS = [
    "how much", "total", "percentage", "percent", "%", "margin",
    "revenue", "net sales", "net income", "operating income", "income",
    "eps", "earnings per share", "earnings",
    "cash", "debt", "compare", "increase", "decrease", "growth", "change",
    "rate", "ratio", "cost", "expense", "profit", "gross margin",
    "spend", "assets", "equity", "shareholders",
]

EXPLANATORY_KEYWORDS = [
    "why", "what caused", "what led", "what drove", "reason", "explain",
    "describe", "what factors", "what risks", "what risk",
]

# question wording -> section label the classifier assigns
SECTION_HINTS = {
    "risk": "Risk Factors",
    "legal proceeding": "Legal Proceedings",
    "lawsuit": "Legal Proceedings",
    "properties": "Properties",
}


class GraphState(TypedDict):
    query: str
    route: str
    retrieved: List[Dict]
    answer: str
    verification: List[Dict]
    model: str


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


def call_gemini_with_retry(client, prompt, max_attempts=6):
    """Returns (response, model_name). Swaps models when one is exhausted."""
    import time
    from google.genai.errors import ClientError, ServerError
    global _current_model_idx

    exhausted = 0  # models that failed during this call
    while exhausted < len(GEMINI_MODELS):
        model = GEMINI_MODELS[_current_model_idx]
        switch = False

        for attempt in range(1, max_attempts + 1):
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                return response, model
            except (ClientError, ServerError) as e:
                msg = str(e)
                if "PerDay" in msg:
                    print(f"  Daily quota reached for {model}.")
                    switch = True
                    break
                if "NOT_FOUND" in msg or "404" in msg:
                    print(f"  {model} unavailable, skipping.")
                    switch = True
                    break
                retryable = (
                    "RESOURCE_EXHAUSTED" in msg
                    or "UNAVAILABLE" in msg
                    or "503" in msg
                )
                if not retryable:
                    raise
                if attempt == max_attempts:
                    print(f"  {model} still failing after {max_attempts} attempts.")
                    switch = True
                    break
                wait = min(10 * 2 ** (attempt - 1), 90)  # 10s, 20s, 40s, 80s, 90s
                print(f"  {model} busy/rate-limited, waiting {wait}s "
                      f"(retry {attempt}/{max_attempts - 1})...")
                time.sleep(wait)

        if switch:
            exhausted += 1
            _current_model_idx = (_current_model_idx + 1) % len(GEMINI_MODELS)
            if exhausted < len(GEMINI_MODELS):
                print(f"  Switching to {GEMINI_MODELS[_current_model_idx]}.")

    raise RuntimeError("All Gemini models are exhausted or unavailable right now.")


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
        query_lower = state["query"].lower()

        companies = detect_companies(state["query"])  # None = no filter

        def allowed(e):
            return companies is None or e["company"] in companies

        text_indices = [i for i, e in enumerate(entries)
                        if e["type"] == "text" and allowed(e)]
        table_indices = [i for i, e in enumerate(entries)
                         if e["type"] == "table" and allowed(e)]

        # collapse identical passages repeated across years' filings
        # (keep the newest filing's copy) so slots aren't wasted
        best = {}
        for i in text_indices:
            key = " ".join(entries[i]["display_text"].split())[:300]
            if key not in best or entries[i]["filing"] > entries[best[key]]["filing"]:
                best[key] = i
        text_indices = list(best.values())

        # plain similarity ranking, with a soft boost for the section the
        # question is about (up to half the slots) instead of a hard filter
        by_score = sorted(text_indices, key=lambda i: -scores[i])
        top_text = by_score[:n_text]
        for hint, section in SECTION_HINTS.items():
            if hint in query_lower:
                preferred = [i for i in by_score if entries[i]["section"] == section][: n_text // 3]
                rest = [i for i in by_score if i not in preferred]
                top_text = preferred + rest[: n_text - len(preferred)]
                break

        top_table = sorted(table_indices, key=lambda i: -scores[i])[:min(n_table, len(table_indices))]

        combined = sorted(top_text + top_table, key=lambda i: -scores[i])
        return {**state, "retrieved": [entries[i] for i in combined]}

    def retrieve_table_focused(state):
        query = state["query"]

        # hybrid BM25 + embedding + label-match search over table rows only
        # (filtered to the company named in the question, if any).
        # Wide pool: the same row appears in up to 3 filings.
        tables = hybrid_table_search(query, k=40, model=embed_model)

        # Years named in the question -> take the best rows from EACH named
        # year's filing (filenames carry the filing date, e.g. AAPL_10-K_2024-11-01).
        years = list(dict.fromkeys(re.findall(r"\b(20\d{2})\b", query)))
        if years:
            per_year = max(2, 6 // len(years))
            picked = []
            for y in years:
                picked += [e for e in tables if f"_{y}-" in e["filing"]][:per_year]
            picked_ids = {id(e) for e in picked}
            rest = [e for e in tables if id(e) not in picked_ids]
            tables = picked + rest
        tables = tables[:6]

        # a couple of prose chunks for context (n_table=0 -> text only)
        text_state = retrieve(state, n_table=0, n_text=2)
        return {**state, "retrieved": tables + text_state["retrieved"]}

    def retrieve_text_focused(state):
        return retrieve(state, n_table=1, n_text=6)

    def generate(state):
        context_block = "\n\n".join(
            f"[Source {i+1}: {e['filing']}, {e['section']}, type: {e['type']}]\n{e['display_text']}"
            for i, e in enumerate(state["retrieved"])
        )
        prompt = f"""You are a financial analyst assistant. Answer the question using ONLY
the sources below. Some sources are tables (rows of numbers) and some are
prose. Cite sources by their number (e.g. "[Source 2]"). When citing a
number from a table, quote it exactly as it appears — do not round it. If
the question asks for a growth rate, percentage change or other calculation,
take the needed figures from the sources, show them, calculate the result,
and say it is calculated. When the question names a fiscal year, prefer
figures from the filing for that same year (e.g. the 2024 10-K for 2024
figures). Only use sources from the company the question asks about; if none
of the sources are from that company, say the sources don't contain the
answer. If the question does not name a company, say which company your
answer is about. If the question names a segment, product line or line item
(for example Services, iPhone, Server products and cloud services), answer
using the row for exactly that item, even if no consolidated or "Total"
figure appears in the sources. If the question does not name a segment or
product line, answer with the company-wide (consolidated or "Total") figure,
never a segment figure. If the sources don't contain enough information to
answer confidently, say so explicitly rather than guessing.

SOURCES:
{context_block}

QUESTION: {state['query']}

ANSWER:"""
        response, model_used = call_gemini_with_retry(gemini_client, prompt)
        return {**state, "answer": response.text, "model": model_used}

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
                          "answer": "", "verification": [], "model": ""})

    print(f"\nRoute chosen: {result['route']}")
    print(f"Model used: {result['model']}")
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