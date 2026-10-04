# find_row.py
import sys
from filinglens_graph import load_index, normalize

_, entries = load_index()
term = " ".join(sys.argv[1:]).lower()
for e in entries:
    if term in e["display_text"].lower() and e["type"] == "text":
        print(f"--- {e['filing']} | {e['section']}")
        print(e["display_text"][:300], "\n")