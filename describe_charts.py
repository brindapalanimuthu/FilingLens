import os, json, glob, base64, time, requests

API_KEY = os.environ["GEMINI_API_KEY"]   # use whatever env var name your other scripts use
MODELS = ["gemini-2.5-flash-lite", "gemini-2.5-flash"]

PROMPT = """You are analyzing an image extracted from an SEC filing.
Return ONLY JSON with these keys:
- is_chart: true if it is a data chart/graph, false for logos, banners or decoration
- title: the chart title (empty string if none)
- chart_type: e.g. "line", "bar", "pie"
- x_axis: what the x axis shows, with first and last labels
- y_axis: what the y axis shows, with units
- series: list of series names from the legend
- description: 3-5 sentences describing the main trend and how the series compare, using words like "approximately". Do not invent exact values.
- approx_values: list of {"series": str, "x": str, "approx_y": number}, read from the lines/bars at each x-axis tick
"""

def describe(img_path):
    b64 = base64.b64encode(open(img_path, "rb").read()).decode()
    body = {
        "contents": [{"parts": [
            {"text": PROMPT},
            {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
        ]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
    }
    for model in MODELS:
        for attempt in range(3):
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                params={"key": API_KEY}, json=body, timeout=120)
            if r.ok:
                txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(txt), model
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"all models failed for {img_path}: {r.status_code} {r.text[:200]}")

os.makedirs("chart_descriptions", exist_ok=True)
for folder in sorted(glob.glob("chart_images/*")):
    base = os.path.basename(folder)               # AAPL_10-K_2025-10-31
    ticker, form, date = base.split("_")
    out = []
    for img in sorted(glob.glob(f"{folder}/*.jpg")):
        info, model = describe(img)
        info.update({"company": ticker, "form_type": form, "filing_date": date,
                     "image_file": os.path.basename(img), "described_by": model})
        out.append(info)
        print(base, os.path.basename(img), "chart" if info.get("is_chart") else "skip", "|", info.get("title", "")[:70])
    json.dump(out, open(f"chart_descriptions/{base}.json", "w"), indent=2)