import json
d = json.load(open("parsed/AAPL_10-K_2024-11-01.json"))
targets = {"Q4": "31,370", "Q6": "56,950", "Q8": "93,736", "Q9": "364,980", "Q10": "29,943"}

def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, f"{path}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, f"{path}[{i}]")
    else:
        yield path, str(o)

for q, num in targets.items():
    print(f"\n=== {q} {num} ===")
    n = 0
    for path, s in walk(d):
        if num in s:
            i = s.index(num)
            print(path, "|", s[max(0, i-150):i+60].replace("\n", " "))
            n += 1
            if n == 2: break
