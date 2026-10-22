import json, pickle

parsed = json.load(open("parsed/MSFT_10-K_2025-07-30.json"))
t71 = next(t for t in parsed["tables"] if t["table_id"] == 71)
print("RAW TABLE 71 (first 25 rows):")
for r in t71["rows"][:25]:
    print("  ", [c for c in r if c.strip()])

entries = pickle.load(open("full_index/entries.pkl", "rb"))
print("\nINDEXED 'operating income' rows, MSFT FY2025 filing:")
for e in entries:
    if (e["company"] == "MSFT" and "2025-07-30" in e["filing"]
            and e["type"] == "table" and "operating income" in e["embed_text"].lower()):
        print("  ", e["section"], "|", e["embed_text"].split(" | ", 2)[-1][:170])