"""
FilingLens — Stage 1: SEC EDGAR Ingestion

Usage:
    python edgar_ingest.py          # defaults to AAPL
    python edgar_ingest.py MSFT     # any ticker
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import os
import sys
import time

YOUR_NAME = "Brinda"
YOUR_EMAIL = "palanimuthubrinda@gmail.com"  # <-- put your real email here
HEADERS = {
    "User-Agent": f"{YOUR_NAME} {YOUR_EMAIL}",
    "Accept-Encoding": "gzip, deflate",
}

OUTPUT_DIR = "filings_raw"
TIMEOUT = 30


def make_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


SESSION = make_session()


def get_cik(ticker: str) -> str:
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = SESSION.get(url, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    ticker = ticker.upper()
    for entry in data.values():
        if entry["ticker"] == ticker:
            return str(entry["cik_str"]).zfill(10)
    raise ValueError(f"Ticker {ticker} not found in SEC ticker list")


def get_filings(cik: str, form_type: str = "10-K", limit: int = 5) -> list:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = SESSION.get(url, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    recent = data["filings"]["recent"]
    results = []
    for i in range(len(recent["form"])):
        if recent["form"][i] == form_type:
            results.append({
                "accessionNumber": recent["accessionNumber"][i],
                "filingDate": recent["filingDate"][i],
                "primaryDocument": recent["primaryDocument"][i],
            })
        if len(results) >= limit:
            break
    return results


def download_filing(cik: str, filing: dict, save_path: str, max_attempts: int = 5):
    accession_nodash = filing["accessionNumber"].replace("-", "")
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik.lstrip('0')}/{accession_nodash}/{filing['primaryDocument']}"
    )
    for attempt in range(1, max_attempts + 1):
        try:
            resp = SESSION.get(url, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            with open(save_path, "wb") as f:
                f.write(resp.content)
            print(f"Saved: {save_path}")
            return
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            wait = 2 ** attempt
            print(f"  Attempt {attempt}/{max_attempts} failed ({e.__class__.__name__}), retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Failed to download {url} after {max_attempts} attempts")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    TICKER = (sys.argv[1] if len(sys.argv) > 1 else "AAPL").upper()
    FORM_TYPE = "10-K"

    print(f"Looking up CIK for {TICKER}...")
    cik = get_cik(TICKER)
    print(f"CIK: {cik}")

    print(f"Fetching recent {FORM_TYPE} filings...")
    filings = get_filings(cik, form_type=FORM_TYPE, limit=3)
    print(json.dumps(filings, indent=2))

    for filing in filings:
        filename = f"{TICKER}_{FORM_TYPE}_{filing['filingDate']}.html"
        save_path = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(save_path):
            print(f"Already have {save_path}, skipping")
            continue
        download_filing(cik, filing, save_path)
        time.sleep(1)  # be polite to SEC