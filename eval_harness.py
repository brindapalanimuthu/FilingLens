"""
FilingLens — Stage 7: Eval harness (v2.3)
Outcomes: correct / wrong / refused. Retrieval hit tracked separately.
Records which Gemini model answered each question.
Resumes from eval_results.json. Flags:
  --fresh        ignore saved results and rerun everything
  --retry-wrong  rerun only questions that were not correct
"""

from filinglens_graph import build_graph, load_index, normalize
from google import genai
from sentence_transformers import SentenceTransformer
import json
import os
import sys
import time
from collections import defaultdict


EVAL_SET_PATH = os.environ.get("EVAL_SET", "eval_set.json")
RESULTS_PATH = os.environ.get("EVAL_RESULTS", "eval_results.json")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
SLEEP_SECONDS = 13  # keeps you under free-tier requests-per-minute

REFUSAL_PHRASES = [
    "do not contain", "does not contain", "don't contain",
    "not enough information", "insufficient information",
    "cannot answer", "can't answer", "unable to answer",
    "cannot be determined", "not provided in the sources",
    "sources do not", "sources don't",
]


def is_refusal(answer):
    a = answer.lower()
    return any(p in a for p in REFUSAL_PHRASES)


def expected_numbers(q):
    if "expected_numbers" in q:
        return q["expected_numbers"]
    if "expected_number" in q:
        return [q["expected_number"]]
    return []


def answer_matches(q, answer):
    """Does the answer contain what we expect (all numbers / any keyword)?"""
    if q["type"] == "numeric":
        norm = normalize(answer)
        return all(normalize(n) in norm for n in expected_numbers(q))
    if q["type"] == "prose":
        low = answer.lower()
        return any(kw.lower() in low for kw in q["expected_keywords"])
    return False


def retrieval_hit(q, retrieved):
    """Were the expected numbers / keywords present in the retrieved sources?
    For derived answers (e.g. growth %), 'retrieval_numbers' lists the input
    figures that must be retrieved instead of the computed result."""
    if q["type"] == "unanswerable":
        return None
    source = " ".join(e["display_text"] for e in retrieved)
    if q["type"] == "numeric":
        norm = normalize(source)
        needed = q.get("retrieval_numbers") or expected_numbers(q)
        return all(normalize(n) in norm for n in needed)
    low = source.lower()
    return any(kw.lower() in low for kw in q["expected_keywords"])


def outcome_for(q, answer):
    refused = is_refusal(answer)
    if q["type"] == "unanswerable":
        return "correct" if refused else "wrong"
    if refused:
        return "refused"
    return "correct" if answer_matches(q, answer) else "wrong"


def verification_pass_rate(result):
    v = result["verification"]
    if not v:
        return None
    return sum(1 for x in v if x["verified"]) / len(v)


def pct(n, d):
    return f"{n}/{d} ({n / d:.0%})" if d else "n/a"


def load_saved_results():
    """Load previous results; ignore entries from the old harness format."""
    if "--fresh" in sys.argv or not os.path.exists(RESULTS_PATH):
        return []
    try:
        saved = json.load(open(RESULTS_PATH))
    except Exception:
        return []
    saved = [r for r in saved if "outcome" in r]
    if "--retry-wrong" in sys.argv:
        saved = [r for r in saved if r["outcome"] == "correct"]
    return saved


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

    with open(EVAL_SET_PATH, "r") as f:
        raw_eval_set = json.load(f)

    # drop duplicate ids (keep the first occurrence)
    seen_ids, eval_set, dupes = set(), [], set()
    for q in raw_eval_set:
        if q["id"] in seen_ids:
            dupes.add(q["id"])
            continue
        seen_ids.add(q["id"])
        eval_set.append(q)
    if dupes:
        print(f"WARNING: duplicate question ids in {EVAL_SET_PATH} "
              f"(second copies skipped): {sorted(dupes)}\n")

    results_log = load_saved_results()
    done_ids = {r["id"] for r in results_log}
    todo = [q for q in eval_set if q["id"] not in done_ids]

    if done_ids:
        print(f"Resuming: {len(done_ids)} already done, {len(todo)} to run.\n")
    else:
        print(f"Running {len(todo)} eval questions...\n")

    crashed = None
    for i, q in enumerate(todo):
        try:
            result = app.invoke({"query": q["question"], "route": "", "retrieved": [],
                                  "answer": "", "verification": [], "model": ""})
        except Exception as e:
            crashed = f"Q{q['id']}: {type(e).__name__}: {str(e)[:200]}"
            print(f"\nStopped early — {crashed}")
            break

        outcome = outcome_for(q, result["answer"])
        r_hit = retrieval_hit(q, result["retrieved"])
        pass_rate = None if outcome == "refused" else verification_pass_rate(result)

        results_log.append({
            "id": q["id"],
            "question": q["question"],
            "type": q["type"],
            "route": result["route"],
            "model": result.get("model", ""),
            "outcome": outcome,
            "retrieval_hit": r_hit,
            "verification_pass_rate": pass_rate,
            "answer": result["answer"],
        })
        results_log.sort(key=lambda r: r["id"])

        tag = {"correct": "PASS", "wrong": "FAIL", "refused": "REFU"}[outcome]
        hit_str = {True: "yes", False: "NO", None: "n/a"}[r_hit]
        rate_str = f"{pass_rate:.0%}" if pass_rate is not None else "n/a"
        print(f"[{tag}] Q{q['id']} ({q['type']}, model={result.get('model', '?')}, "
              f"retrieved={hit_str}, verified={rate_str}): {q['question']}")

        with open(RESULTS_PATH, "w") as f:
            json.dump(results_log, f, indent=2)

        if i < len(todo) - 1:
            time.sleep(SLEEP_SECONDS)

    # ---------------- summary ----------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total = len(results_log)
    if total == 0:
        print("No results yet.")
        if crashed:
            print(f"Run incomplete: {crashed}")
        sys.exit(0)

    counts = defaultdict(int)
    for r in results_log:
        counts[r["outcome"]] += 1
    print(f"Overall:  correct {pct(counts['correct'], total)} | "
          f"wrong {counts['wrong']} | refused {counts['refused']}")

    print("\nBy type:")
    for t in ("numeric", "prose", "unanswerable"):
        rows = [r for r in results_log if r["type"] == t]
        if rows:
            ok = sum(1 for r in rows if r["outcome"] == "correct")
            print(f"  {t:13} {pct(ok, len(rows))}")

    by_model = defaultdict(list)
    for r in results_log:
        by_model[r.get("model") or "unknown (earlier run)"].append(r)
    if len(by_model) > 1 or "unknown (earlier run)" not in by_model:
        print("\nBy model:")
        for m, rows in by_model.items():
            ok = sum(1 for r in rows if r["outcome"] == "correct")
            print(f"  {m:24} {pct(ok, len(rows))}")

    answerable = [r for r in results_log if r["retrieval_hit"] is not None]
    if answerable:
        hits = sum(1 for r in answerable if r["retrieval_hit"])
        print(f"\nRetrieval hit rate: {pct(hits, len(answerable))} "
              f"(expected answer present in retrieved sources)")
        gen_fail = [r for r in answerable if r["retrieval_hit"] and r["outcome"] != "correct"]
        ret_fail = [r for r in answerable if not r["retrieval_hit"] and r["outcome"] != "correct"]
        print(f"Failures due to retrieval: {len(ret_fail)} | due to generation: {len(gen_fail)}")

    rates = [r["verification_pass_rate"] for r in results_log
             if r["verification_pass_rate"] is not None]
    if rates:
        print(f"Numeric verification (non-refused answers): "
              f"{sum(rates) / len(rates):.0%} avg across {len(rates)} answers")

    bad = [r for r in results_log if r["outcome"] != "correct"]
    if bad:
        print("\nNot correct:")
        for r in bad:
            print(f"  Q{r['id']} [{r['outcome']}, retrieved={r['retrieval_hit']}]: {r['question']}")
            print(f"    Got: {r['answer'][:150]}")

    if crashed:
        print(f"\nRun incomplete ({total}/{len(eval_set)} done): {crashed}")
    print(f"\nFull results saved to {RESULTS_PATH}")