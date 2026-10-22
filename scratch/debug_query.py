import sys
from sentence_transformers import SentenceTransformer
from hybrid_search import hybrid_table_search

q = "What was Microsoft's net income in fiscal 2025?"
model = SentenceTransformer("all-MiniLM-L6-v2")

print("TOP 12 TABLE ROWS:\n")
for i, e in enumerate(hybrid_table_search(q, k=12, model=model), 1):
    print(f"[{i}] {e['display_text'][:220]}\n")