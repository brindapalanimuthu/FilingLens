import json

data = json.load(open("eval_set.json"))
seen, clean = set(), []
for q in data:
    key = (q["id"], q["question"])
    if key in seen:
        continue
    seen.add(key)
    clean.append(q)

ids = [q["id"] for q in clean]
assert len(ids) == len(set(ids)), f"still duplicate ids: {ids}"
json.dump(clean, open("eval_set.json", "w"), indent=1)
print(f"{len(data)} -> {len(clean)} questions")