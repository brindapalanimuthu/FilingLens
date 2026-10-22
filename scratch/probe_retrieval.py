from sentence_transformers import SentenceTransformer
from filinglens_graph import build_graph, load_index

QUESTIONS = [
    "How did Microsoft's total revenue change from fiscal 2024 to fiscal 2025?",
    "What was Microsoft's operating income in fiscal 2025?",
    "What risks does Microsoft describe related to AI?",
    "By what percentage did Apple's total net sales grow from 2023 to 2024?",
]


class _Resp:
    text = "(skipped)"


class _Models:
    def generate_content(self, model, contents):
        return _Resp()


class FakeClient:
    models = _Models()


embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings, entries = load_index()
app = build_graph(embed_model, embeddings, entries, FakeClient())

for q in QUESTIONS:
    r = app.invoke({"query": q, "route": "", "retrieved": [],
                    "answer": "", "verification": [], "model": ""})
    print("=" * 70)
    print("Q:", q)
    print("route:", r["route"])
    for i, e in enumerate(r["retrieved"], 1):
        print(f"  [{i}] {e['filing'][:4]}{e['filing'][-10:-6]} | {e['section'][:32]:32} | "
              f"{e['display_text'][-110:] if e['type'] == 'table' else e['display_text'][:110]}")