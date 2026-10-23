import json, glob, os, time, requests

UA = {"User-Agent": "Brinda Palanimuthu your_email@example.com"}  # use the same UA as edgar_ingest.py
CIK = {"AAPL": "0000320193"}

subs = {}
for t, cik in CIK.items():
    subs[t] = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=UA).json()["filings"]["recent"]
    time.sleep(0.2)

for f in sorted(glob.glob("parsed/AAPL_*.json")):
    base = os.path.basename(f)[:-5]                     # AAPL_10-K_2023-11-03
    ticker, form, date = base.split("_")
    d = json.load(open(f))
    if not d.get("images"):
        continue
    r = subs[ticker]
    idx = next((i for i in range(len(r["form"]))
                if r["form"][i] == form and r["filingDate"][i] == date), None)
    if idx is None:
        print("no match for", base); continue
    acc = r["accessionNumber"][idx].replace("-", "")
    out = f"chart_images/{base}"
    os.makedirs(out, exist_ok=True)
    for im in d["images"]:
        url = f"https://www.sec.gov/Archives/edgar/data/{int(CIK[ticker])}/{acc}/{im['src']}"
        resp = requests.get(url, headers=UA)
        if resp.ok:
            open(f"{out}/{im['src']}", "wb").write(resp.content)
            print("saved", out, im["src"], len(resp.content), "bytes")
        else:
            print("FAILED", url, resp.status_code)
        time.sleep(0.2)