import os, time
from google import genai
from sentence_transformers import SentenceTransformer
from filinglens_graph import build_graph, load_index

QUESTIONS = [
    "How did Microsoft's total revenue change from fiscal 2024 to fiscal 2025?",
    "Compare Apple's and Microsoft's net income in 2024.",
    "What was total net sales in 2024?",
    "What was Microsoft's operating income in fiscal 2025?",
    "What drove growth in Microsoft's Intelligent Cloud segment in fiscal 2025?",
    "What risks does Microsoft describe related to AI?",
    "By what percentage did Apple's total net sales grow from 2023 to 2024?",
    "What was Microsoft's Azure revenue in fiscal 2025?",
]

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings, entries = load_index()
app = build_graph(embed_model, embeddings, entries, client)

for q in QUESTIONS:
    r = app.invoke({"query": q, "route": "", "retrieved": [],
                    "answer": "", "verification": [], "model": ""})
    print("=" * 70)
    print("Q:", q)
    print("route:", r["route"], "| model:", r["model"])
    print("sources:", ", ".join(f"{e['filing'][:4]}{e['filing'][-10:-6]}" for e in r["retrieved"]))
    print("A:", r["answer"].strip()[:600])
    bad = [v["number"] for v in r["verification"] if not v["verified"]]
    print("unverified numbers:", bad or "none")
    time.sleep(13)