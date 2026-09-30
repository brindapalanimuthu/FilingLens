import json
from table_chunker import table_to_row_chunks

d = json.load(open("parsed/AAPL_10-K_2024-11-01.json"))
for i in (14, 16):
    print(f"\n--- table {i} ---")
    for c in table_to_row_chunks(d["tables"][i], "AAPL_10-K_2024")[:6]:
        print(c["text"])
