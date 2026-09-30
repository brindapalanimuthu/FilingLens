import json, pickle, re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

STOP = set("what was were is the a an of in as for did how much apple apples "
           "s did do does to and on at by".split())

def tok(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP]

model = SentenceTransformer("all-MiniLM-L6-v2")
emb = np.load("full_index/embeddings.npy")
emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
entries = pickle.load(open("full_index/entries.pkl", "rb"))

tidx = [i for i, e in enumerate(entries) if e["type"] == "table"]
t_emb = emb[tidx]
bm25 = BM25Okapi([tok(entries[i]["embed_text"]) for i in tidx])

def rank_of(order, num):
    for r, j in enumerate(order[:200], 1):
        if num in entries[tidx[j]]["display_text"]:
            return r
    return None

def rrf(*orders, k=60):
    score = {}
    for order in orders:
        for r, j in enumerate(order, 1):
            score[j] = score.get(j, 0) + 1 / (k + r)
    return sorted(score, key=score.get, reverse=True)

evals = [q for q in json.load(open("eval_set.json")) if q["type"] == "numeric"]
stats = {"embed": [], "bm25": [], "hybrid": []}

for q in evals:
    qv = model.encode([q["question"]])[0]
    qv /= np.linalg.norm(qv)
    o_emb = list(np.argsort(-(t_emb @ qv)))
    o_bm = list(np.argsort(-bm25.get_scores(tok(q["question"]))))
    o_hy = rrf(o_emb[:100], o_bm[:100])
    r = {"embed": rank_of(o_emb, q["expected_number"]),
         "bm25": rank_of(o_bm, q["expected_number"]),
         "hybrid": rank_of(o_hy, q["expected_number"])}
    for k, v in r.items():
        stats[k].append(v)
    print(f"Q{q['id']} {q['expected_number']:>8}  embed={r['embed']}  bm25={r['bm25']}  hybrid={r['hybrid']}")

n = len(evals)
for k, ranks in stats.items():
    for cut in (5, 10):
        hits = sum(1 for x in ranks if x and x <= cut)
        print(f"{k:7} recall@{cut}: {hits}/{n}")