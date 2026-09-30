"""
FilingLens — Stage 7: Eval harness
"""

from filinglens_graph import build_graph, load_index, normalize
from google import genai
from sentence_transformers import SentenceTransformer
import json
import os
import sys
import time


EVAL_SET_PATH = "eval_set.json"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def score_question(question, result):
    answer_lower = result["answer"].lower()
    if question["type"] == "numeric":
        expected = normalize(question["expected_number"])
        answer_normalized = normalize(result["answer"])
        return expected in answer_normalized
    else:
        return any(kw.lower() in answer_lower for kw in question["expected_keywords"])


def verification_pass_rate(result):
    verification = result["verification"]
    if not verification:
        return None
    passed = sum(1 for v in verification if v["verified"])
    return passed / len(verification)


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
        eval_set = json.load(f)

    print(f"Running {len(eval_set)} eval questions...\n")

    results_log = []
    for q in eval_set:
        result = app.invoke({"query": q["question"], "route": "", "retrieved": [],
                              "answer": "", "verification": []})
        correct = score_question(q, result)
        pass_rate = verification_pass_rate(result)

        results_log.append({
            "id": q["id"],
            "question": q["question"],
            "type": q["type"],
            "route": result["route"],
            "correct": correct,
            "verification_pass_rate": pass_rate,
            "answer": result["answer"],
        })

        status = "PASS" if correct else "FAIL"
        rate_str = f"{pass_rate:.0%}" if pass_rate is not None else "n/a"
        print(f"[{status}] Q{q['id']} ({q['type']}, route={result['route']}, "
              f"verified={rate_str}): {q['question']}")

        with open("eval_results.json", "w") as f:
            json.dump(results_log, f, indent=2)

        if q is not eval_set[-1]:
            time.sleep(13)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    n_correct = sum(1 for r in results_log if r["correct"])
    accuracy = n_correct / len(results_log)
    print(f"Accuracy: {n_correct}/{len(results_log)} ({accuracy:.0%})")

    rates = [r["verification_pass_rate"] for r in results_log if r["verification_pass_rate"] is not None]
    if rates:
        avg_pass_rate = sum(rates) / len(rates)
        print(f"Avg verification pass rate: {avg_pass_rate:.0%} "
              f"(across {len(rates)} questions with numeric claims)")

    n_failed = [r for r in results_log if not r["correct"]]
    if n_failed:
        print(f"\nFailed questions:")
        for r in n_failed:
            print(f"  Q{r['id']}: {r['question']}")
            print(f"    Got: {r['answer'][:150]}")

    print(f"\nFull results saved to eval_results.json")
