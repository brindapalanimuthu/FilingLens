import json

NEW = [
    {"question": "What was Apple's iPhone net sales in 2024?",
     "type": "numeric", "expected_number": "201,183"},
    {"question": "What were Apple's Services net sales in 2024?",
     "type": "numeric", "expected_number": "96,169"},
    {"question": "What were Apple's total net sales in 2024 and 2023?",
     "type": "numeric", "expected_numbers": ["391,035", "383,285"]},
    {"question": "What was Microsoft's net income in fiscal 2025 and fiscal 2024?",
     "type": "numeric", "expected_numbers": ["101,832", "88,136"]},
    {"question": "By what percentage did Microsoft's total revenue grow from fiscal 2024 to fiscal 2025?",
     "type": "numeric", "expected_number": "14.9"},
    {"question": "What was Microsoft's Server products and cloud services revenue in fiscal 2024?",
     "type": "numeric", "expected_number": "97,726"},
    {"question": "What risks does Apple cite related to its supply chain?",
     "type": "prose", "expected_keywords": ["single-source", "outsourcing partners"]},
    {"question": "What was Apple's net income in fiscal 2030?",
     "type": "unanswerable"},
]

norm = lambda s: "".join(s.lower().split())
qs = json.load(open("eval_set.json"))
have = {norm(q["question"]) for q in qs}
next_id = max(q["id"] for q in qs) + 1
added = 0
for n in NEW:
    if norm(n["question"]) in have:
        continue
    qs.append({"id": next_id, **n})
    next_id += 1
    added += 1
json.dump(qs, open("eval_set.json", "w"), indent=2)
print(f"added {added}, total {len(qs)}")