from sentence_transformers import SentenceTransformer
from hybrid_search import hybrid_table_search, split_parts

m = SentenceTransformer("all-MiniLM-L6-v2")
q = "What were Apple's iPhone and Services net sales in 2025?"
print("parts:", split_parts(q))
for i, e in enumerate(hybrid_table_search(q, k=40, model=m)[:12], 1):
    print(i, e["filing"][9:19], "|", e["display_text"][80:170])