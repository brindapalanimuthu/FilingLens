import pickle, re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

STOP = set("what was were is the a an of in as for did how much apple apples "
           "s did do does to and on at by".split())

def tok(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP]

_model = None
_state = {}

def _load(model=None):
    global _model
    if _state:
        return
    _model = model or SentenceTransformer("all-MiniLM-L6-v2")
    emb = np.load("full_index/embeddings.npy")
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    entries = pickle.load(open("full_index/entries.pkl", "rb"))
    tidx = [i for i, e in enumerate(entries) if e["type"] == "table"]
    _state.update(
        entries=entries, tidx=tidx, t_emb=emb[tidx],
        bm25=BM25Okapi([tok(entries[i]["embed_text"]) for i in tidx]),
    )

def _rrf(*orders, k=60):
    score = {}
    for order in orders:
        for r, j in enumerate(order, 1):
            score[j] = score.get(j, 0) + 1 / (k + r)
    return sorted(score, key=score.get, reverse=True)

def hybrid_table_search(query, k=8, model=None):
    """Return top-k table-row entries using BM25 + embedding fusion."""
    _load(model)
    qv = _model.encode([query])[0]
    qv = qv / np.linalg.norm(qv)
    o_emb = list(np.argsort(-(_state["t_emb"] @ qv)))[:100]
    o_bm = list(np.argsort(-_state["bm25"].get_scores(tok(query))))[:100]
    top = _rrf(o_emb, o_bm)[:k]
    return [_state["entries"][_state["tidx"][j]] for j in top]