import json, glob
evals = json.load(open("eval_set.json"))
blobs = {f: json.dumps(json.load(open(f))) for f in glob.glob("parsed/*.json")}
for q in evals:
    if q["type"] != "numeric":
        continue
    hits = [f for f, b in blobs.items() if q["expected_number"] in b]
    print(q["id"], q["expected_number"], "->", len(hits), "files:", hits)
