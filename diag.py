import json, glob, statistics

for path in sorted(glob.glob("parsed/MSFT_*.json")):
    text = json.load(open(path))["text"]
    lens = sorted(len(t) for t in text)
    print(path)
    print(f"  chunks={len(text)}  median={statistics.median(lens)}  max={lens[-1]}  >5000 chars: {sum(l > 5000 for l in lens)}")
    longest = max(text, key=len)
    print(f"  longest starts: {longest[:150]!r}\n")