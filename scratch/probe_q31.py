from types import SimpleNamespace
from sentence_transformers import SentenceTransformer
from filinglens_graph import build_graph, load_index


class StubModels:
    def generate_content(self, model, contents):
        return SimpleNamespace(text="stub")


client = SimpleNamespace(models=StubModels())
embed = SentenceTransformer("all-MiniLM-L6-v2")
embeddings, entries = load_index()
app = build_graph(embed, embeddings, entries, client)

q = "What risks does Apple cite related to its supply chain?"
kws = ["single-source", "single or limited", "outsourcing partners"]
r = app.invoke({"query": q, "route": "", "retrieved": [], "answer": "",
                "verification": [], "model": ""})
print("route:", r["route"])
for i, e in enumerate(r["retrieved"], 1):
    low = e["display_text"].lower()
    hit = [k for k in kws if k in low]
    print(f"{i} {e['filing'][9:19]} | {e['section'][:22]:22} | hits={hit} | {e['display_text'][:80]}")