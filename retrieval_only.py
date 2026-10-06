import json
import os
from types import SimpleNamespace
from sentence_transformers import SentenceTransformer
from filinglens_graph import build_graph, load_index
from eval_harness import retrieval_hit


class StubModels:
    def generate_content(self, model, contents):
        return SimpleNamespace(text="stub")


EVAL_SET = os.environ.get("EVAL_SET", "eval_set.json")

client = SimpleNamespace(models=StubModels())
embed = SentenceTransformer("all-MiniLM-L6-v2")
embeddings, entries = load_index()
app = build_graph(embed, embeddings, entries, client)

total = misses = 0
for q in json.load(open(EVAL_SET)):
    if q["type"] == "unanswerable":
        continue
    r = app.invoke({"query": q["question"], "route": "", "retrieved": [],
                    "answer": "", "verification": [], "model": ""})
    total += 1
    if not retrieval_hit(q, r["retrieved"]):
        misses += 1
        print("MISS", q["id"], q["question"])
print(f"[{EVAL_SET}] retrieval hits: {total - misses}/{total}")