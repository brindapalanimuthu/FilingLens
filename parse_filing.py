"""
FilingLens — Stage 2: Parsing
"""

from bs4 import BeautifulSoup
import json
import os
import glob


INPUT_DIR = "filings_raw"
OUTPUT_DIR = "parsed"


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
        src = img.get("src", "")
        alt = img.get("alt", "")
        chunks["images"].append({"image_id": i, "src": src, "alt": alt})

    for table in soup.find_all("table"):
        table.decompose()

    for hidden in soup.find_all(style=lambda v: v and "display:none" in v.replace(" ", "").lower()):
        hidden.decompose()
    for hidden in soup.find_all("ix:hidden"):
        hidden.decompose()

    paragraphs = []
    for p in soup.find_all(["p", "div", "span"]):
        text = p.get_text(separator=" ", strip=True)
        if text and len(text) > 40:
            paragraphs.append(text)

    seen = set()
    exact_deduped = []
    for text in paragraphs:
        if text not in seen:
            seen.add(text)
            exact_deduped.append(text)

    by_length_desc = sorted(exact_deduped, key=len, reverse=True)
    kept = []
    for text in by_length_desc:
        if not any(text != longer and text in longer for longer in kept):
            kept.append(text)

    order = {text: i for i, text in enumerate(exact_deduped)}
    kept.sort(key=lambda t: order[t])
    chunks["text"] = kept

    return chunks


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    html_files = glob.glob(os.path.join(INPUT_DIR, "*.html"))
    print(f"Found {len(html_files)} filing(s) to parse")

    for html_path in html_files:
        print(f"\nParsing {html_path}...")
        chunks = parse_filing(html_path)

        print(f"  Tables found: {len(chunks['tables'])}")
        print(f"  Images found: {len(chunks['images'])}")
        print(f"  Text chunks found: {len(chunks['text'])}")

        basename = os.path.splitext(os.path.basename(html_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{basename}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)
        print(f"  Saved: {out_path}")
