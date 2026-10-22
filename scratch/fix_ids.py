import json
qs = json.load(open("eval_set.json"))
seen, max_id = set(), max(q["id"] for q in qs)
for q in qs:
    if q["id"] in seen:
        max_id += 1
        q["id"] = max_id
    seen.add(q["id"])
    q["question"] = (q["question"]
        .replace("effectivetax", "effective tax")
        .replace("andcash", "and cash"))
json.dump(qs, open("eval_set.json", "w"), indent=2)
print(len(qs), "questions, ids unique:", len({q["id"] for q in qs}) == len(qs))