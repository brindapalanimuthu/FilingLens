from sentence_transformers import SentenceTransformer
from hybrid_search import hybrid_table_search

m = SentenceTransformer("all-MiniLM-L6-v2")
q = "What was Apple's iPhone net sales in 2024?"
for i, e in enumerate(hybrid_table_search(q, k=40, model=m)[:12], 1):
    print(i, e["filing"], "|", e["display_text"][:110])