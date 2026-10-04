"""
FilingLens — Stage 2: Parsing (v2: leaf-level chunks)

Usage:
    python parse_filing.py          # parse every filing not yet parsed
    python parse_filing.py MSFT     # only filings for this ticker
    python parse_filing.py --force  # reparse (with a ticker, only that ticker)
"""

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import json
import os
import glob
import re
import sys
import warnings

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

INPUT_DIR = "filings_raw"
OUTPUT_DIR = "parsed"
MIN_CHARS = 40
MAX_CHARS = 1500


def split_long(text: str, max_chars: int = MAX_CHARS) -> list:
    """Split overly long text at sentence boundaries."""
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) + 1 > max_chars:
            pieces.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        pieces.append(cur)
    return pieces


def parse_filing(html_path: str) -> dict:
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "lxml")

    chunks = {"tables": [], "images": [], "text": []}

    for i, table in enumerate(soup.find_all("table")):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if any(cells):
                rows.append(cells)
        if rows:
            chunks["tables"].append({"table_id": i, "rows": rows})

    for i, img in enumerate(soup.find_all("img")):
        chunks["images"].append({
            "image_id": i,
            "src": img.get("src", ""),
            "alt": img.get("alt", ""),
        })

    for table in soup.find_all("table"):
        table.decompose()

    for hidden in soup.find_all(style=lambda v: v and "display:none" in v.replace(" ", "").lower()):
        hidden.decompose()
    for hidden in soup.find_all("ix:hidden"):
        hidden.decompose()

    # leaf-level blocks only: no nested <p>/<div> inside
    texts = []
    for el in soup.find_all(["p", "div"]):
        if el.find(["p", "div"]) is not None:
            continue
        text = el.get_text(separator=" ", strip=True)
        if len(text) > MIN_CHARS:
            texts.extend(split_long(text))

    # exact-duplicate removal, keep first occurrence in document order
    seen, final = set(), []
    for t in texts:
        if t not in seen:
            seen.add(t)
            final.append(t)

    chunks["text"] = final
    return chunks


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    force = "--force" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ticker = args[0].upper() if args else None

    pattern = f"{ticker}_*.html" if ticker else "*.html"
    html_files = sorted(glob.glob(os.path.join(INPUT_DIR, pattern)))
    print(f"Found {len(html_files)} filing(s)")

    for html_path in html_files:
        basename = os.path.splitext(os.path.basename(html_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{basename}.json")

        if os.path.exists(out_path) and not force:
            print(f"Already parsed {out_path}, skipping")
            continue

        print(f"\nParsing {html_path}...")
        chunks = parse_filing(html_path)
        lens = sorted(len(t) for t in chunks["text"])
        print(f"  Tables: {len(chunks['tables'])} | Images: {len(chunks['images'])} | "
              f"Text chunks: {len(chunks['text'])} (median {lens[len(lens)//2] if lens else 0}, max {lens[-1] if lens else 0})")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)
        print(f"  Saved: {out_path}")