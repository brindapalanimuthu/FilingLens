import pickle

entries = pickle.load(open("full_index/entries.pkl", "rb"))
msft25 = [e for e in entries
          if e["company"] == "MSFT" and e["type"] == "table"
          and "2025-07-30" in e["filing"]]
print("MSFT FY25 table rows:", len(msft25))

hits = [e for e in msft25 if "net income" in e["embed_text"].lower()]
print("rows containing 'net income':", len(hits))
for e in hits[:10]:
    print(" ", e["section"], "|", e["embed_text"][:230])

print("\nrows containing 'total revenue':")
for e in [e for e in msft25 if "total revenue" in e["embed_text"].lower()][:5]:
    print(" ", e["section"], "|", e["embed_text"][:230])