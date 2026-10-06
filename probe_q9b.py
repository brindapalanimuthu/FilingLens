from sentence_transformers import SentenceTransformer
from hybrid_search import hybrid_table_search, split_parts

m = SentenceTransformer("all-MiniLM-L6-v2")
q = "What were Apple's iPhone and Services net sales in 2025?"
for part in split_parts(q):
    print("\nPART:", part)
    rows = hybrid_table_search(part, k=40, model=m, companies={"AAPL"})
    rows = [e for e in rows if "_2025-" in e["filing"]]
    for i, e in enumerate(rows[:8], 1):
        print(i, e["display_text"][60:150])