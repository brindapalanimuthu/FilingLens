import json, pickle
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
emb = np.load("full_index/embeddings.npy")
emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
entries = pickle.load(open("full_index/entries.pkl", "rb"))
evals = [q for q in json.load(open("eval_set.json")) if q["type"] == "numeric"]

hit_at = {5: 0, 10: 0, 20: 0}
for q in evals:
    qv = model.encode([q["question"]])[0]
    qv = qv / np.linalg.norm(qv)
    order = np.argsort(-(emb @ qv))
    rank = None
    for r, idx in enumerate(order[:200], 1):
        e = entries[idx]
        if e["type"] == "table" and q["expected_number"] in e["display_text"]:
            rank = r
            break
    for k in hit_at:
        if rank and rank <= k:
            hit_at[k] += 1
    print(f"Q{q['id']} expected {q['expected_number']}: first matching table row at rank {rank}")
    for idx in order[:3]:
        print("    top:", entries[idx]["type"], "|", entries[idx]["embed_text"][:110])

n = len(evals)
print(f"\nrecall@5={hit_at[5]}/{n}  @10={hit_at[10]}/{n}  @20={hit_at[20]}/{n}")