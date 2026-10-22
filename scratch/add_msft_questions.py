import json

NEW = [
    {"question": "What was Microsoft's total revenue in fiscal 2025?",
     "type": "numeric", "expected_number": "281,724"},
    {"question": "What was Microsoft's total revenue in fiscal 2024?",
     "type": "numeric", "expected_number": "245,122"},
    {"question": "What was Microsoft's net income in fiscal 2024?",
     "type": "numeric", "expected_number": "88,136"},
    {"question": "What was Microsoft's net income in fiscal 2026?",
     "type": "numeric", "expected_number": "133,749"},
    {"question": "What was Microsoft's stock-based compensation expense in fiscal 2025?",
     "type": "numeric", "expected_number": "11,974"},
    {"question": "What was Microsoft's total revenue and net income in fiscal 2025?",
     "type": "numeric", "expected_numbers": ["281,724", "101,832"]},
    {"question": "What was Microsoft's iPhone revenue in fiscal 2025?",
     "type": "unanswerable"},
    {"question": "What was Apple's Azure revenue in 2024?",
     "type": "unanswerable"},
]

qs = json.load(open("eval_set.json"))
have = {q["question"] for q in qs}
next_id = max(q["id"] for q in qs) + 1
added = 0
for n in NEW:
    if n["question"] in have:
        continue
    qs.append({"id": next_id, **n})
    next_id += 1
    added += 1
json.dump(qs, open("eval_set.json", "w"), indent=2)
print(f"added {added}, total {len(qs)}")