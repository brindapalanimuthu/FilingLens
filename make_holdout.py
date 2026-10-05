import json

Q = [
    ("What was Apple's iPhone net sales in 2025?", ["209,586"]),
    ("What were Apple's Services net sales in 2025?", ["109,158"]),
    ("What was Apple's total net sales in 2025?", ["416,161"]),
    ("What was Apple's Services gross margin percentage in 2025?", ["75.4"]),
    ("What was Microsoft's Server products and cloud services revenue in fiscal 2026?", ["129,425"]),
    ("What was Microsoft's Dynamics products and cloud services revenue in fiscal 2025?", ["7,827"]),
    ("What was Microsoft's Enterprise and partner services revenue in fiscal 2026?", ["8,260"]),
    ("What was Microsoft's Microsoft 365 Commercial products and cloud services revenue in fiscal 2025?", ["87,767"]),
    ("What were Apple's iPhone and Services net sales in 2025?", ["209,586", "109,158"]),
]
UNANSWERABLE = [
    "What was Apple's total net sales in fiscal 2026?",
    "What was Microsoft's Nintendo revenue in fiscal 2025?",
]

out, n = [], 1
for text, nums in Q:
    d = {"id": n, "question": text, "type": "numeric"}
    d["expected_numbers" if len(nums) > 1 else "expected_number"] = nums if len(nums) > 1 else nums[0]
    out.append(d)
    n += 1
for text in UNANSWERABLE:
    out.append({"id": n, "question": text, "type": "unanswerable"})
    n += 1

json.dump(out, open("eval_holdout.json", "w"), indent=2)
print(len(out), "questions written")