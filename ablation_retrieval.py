"""
FilingLens — retrieval ablation (no LLM calls, no Gemini quota).

For every answerable eval question, checks whether the numbers needed to answer it
appear in the top-6 table rows that would be sent to the LLM, comparing:
  - Embeddings only
  - BM25 only
  - Hybrid (embeddings + BM25 + label match + re-ranking)   <- FilingLens

All three get the same company / form-type filter, the same candidate pool (40),
and the same fiscal-year picking step as filinglens_graph.py, so the only thing
that differs is the ranking method.

Usage:  python ablation_retrieval.py
"""
import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import hybrid_search as h

SETS = [
    ("Main (10-K)", "eval_set.json"),
    ("Held-out (10-K)", "eval_holdout.json"),
    ("10-Q", "eval_10q.json"),
]
POOL, TOP = 40, 6


def norm(s):
    return re.sub(r"[,$%()]", "", s)


def needed_numbers(q):
    if q.get("type") == "unanswerable":
        return None
    nums = (q.get("retrieval_numbers") or q.get("expected_numbers")
            or ([q["expected_number"]] if q.get("expected_number") else None))
    return nums


def in_fiscal_year(entry, year):                 # same rule as filinglens_graph.py
    fy = entry.get("fiscal_year")
    return fy == int(year) if fy else f"_{year}-" in entry["filing"]


def pick(pool, query):                           # same year-picking as filinglens_graph.py
    years = list(dict.fromkeys(re.findall(r"\b(20\d{2})\b", query)))
    tables = pool
    if years:
        per_year = max(2, 6 // len(years))
        picked = []
        for y in years:
            picked += [e for e in tables if in_fiscal_year(e, y)][:per_year]
        ids = {id(e) for e in picked}
        tables = picked + [e for e in tables if id(e) not in ids]
    return tables[:TOP]


def allowed_mask(query):
    st = h._state
    comps = h.detect_companies(query)
    allowed = (np.isin(st["t_company"], list(comps)) if comps
               else np.ones(len(st["tidx"]), dtype=bool))
    return allowed & (st["t_form"] == h.detect_form_type(query))


def to_entries(js):
    st = h._state
    return [st["entries"][st["tidx"][j]] for j in js]


def embed_only(query, model):
    st = h._state
    qv = model.encode([h.expand_terms(query)])[0]
    qv = qv / np.linalg.norm(qv)
    allowed = allowed_mask(query)
    order = [j for j in np.argsort(-(st["t_emb"] @ qv)) if allowed[j]][:POOL]
    return to_entries(order)


def bm25_only(query):
    st = h._state
    bm = st["bm25"].get_scores(h.tok(h.expand_terms(query)))
    allowed = allowed_mask(query)
    order = [j for j in np.argsort(-bm) if allowed[j]][:POOL]
    return to_entries(order)


if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")
    h._load(model)

    methods = {
        "Embeddings only": lambda q: embed_only(q, model),
        "BM25 only": bm25_only,
        "Hybrid (FilingLens)": lambda q: h.hybrid_table_search(q, k=POOL, model=model),
    }

    results = {m: {} for m in methods}
    misses = {m: [] for m in methods}
    for set_name, path in SETS:
        qs = [q for q in json.load(open(path)) if needed_numbers(q)]
        for m, fn in methods.items():
            hits = 0
            for q in qs:
                rows = pick(fn(q["question"]), q["question"])
                text = norm(" ".join(e["display_text"] for e in rows))
                if all(norm(n) in text for n in needed_numbers(q)):
                    hits += 1
                else:
                    misses[m].append(f"[{set_name}] {q['question']}")
            results[m][set_name] = (hits, len(qs))

    names = [n for n, _ in SETS]
    print("\n| Retrieval method | " + " | ".join(names) + " | Overall |")
    print("|---|" + "---|" * (len(names) + 1))
    for m in methods:
        cells = [f"{results[m][n][0]} / {results[m][n][1]}" for n in names]
        th = sum(results[m][n][0] for n in names)
        tn = sum(results[m][n][1] for n in names)
        print(f"| {m} | " + " | ".join(cells) + f" | **{th} / {tn}** ({100*th//tn}%) |")

    print("\nQuestions each method missed:")
    for m in methods:
        print(f"\n{m}: {len(misses[m])} missed")
        for line in misses[m]:
            print("  -", line)
    json.dump({"results": results, "misses": misses},
              open("ablation_results.json", "w"), indent=2)
    print("\nSaved ablation_results.json")