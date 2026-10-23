"""
FilingLens — Stage 5: Combined text + table + chart index (multi-company, 10-K + 10-Q)

Reads section labels from classified_deberta/ when a file exists there,
otherwise falls back to classified/. Every entry gets `company` (ticker),
`fiscal_year`, `form_type` ('10-K' / '10-Q') and `fiscal_quarter`
(1-3 for 10-Qs, 0 for 10-Ks).

Chart entries (type='chart'): for filings whose chart_descriptions/ JSON flags
an image as a chart, the exact numbers come from the filing's own stock
performance table (not from the vision model), so they can be verified.
"""

from sentence_transformers import SentenceTransformer
import datetime as dt
import json
import os
import glob
import re
import numpy as np
import pickle
from table_chunker import table_to_row_chunks


# classified/ first, then classified_deberta/ overrides it per filing
INPUT_DIRS = ["classified", "classified_deberta"]
OUTPUT_DIR = "full_index"
CHART_DIR = "chart_descriptions"
MODEL_NAME = "all-MiniLM-L6-v2"

TICKER_NAMES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
}
FISCAL_YEAR_END = {"AAPL": 9, "MSFT": 6}   # month the fiscal year ends


def company_info(filing_name):
    """'MSFT_10-Q_2024-10-30' -> ('MSFT', 'Microsoft Corporation', 2025, '10-Q', 1).

    10-K: fiscal year = filing year (holds for Apple, Sept FY filed Oct/Nov,
    and Microsoft, June FY filed July).
    10-Q: filed ~5 weeks after quarter end, so quarter end = filing date - 35 days;
    fiscal quarter and fiscal year are derived from the company's FY-end month."""
    ticker = filing_name.split("_")[0].upper()
    fm = re.search(r"10-([KQ])", filing_name)
    form_type = f"10-{fm.group(1)}" if fm else "10-K"
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", filing_name)
    fiscal_year, fiscal_quarter = None, 0
    if m:
        filed = dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        fiscal_year = filed.year
        fye = FISCAL_YEAR_END.get(ticker)
        if form_type == "10-Q" and fye:
            qend = filed - dt.timedelta(days=35)
            fiscal_quarter = ((qend.month - fye) % 12) // 3
            fiscal_year = qend.year + 1 if qend.month > fye else qend.year
    return (ticker, TICKER_NAMES.get(ticker, ticker),
            fiscal_year, form_type, fiscal_quarter)


def is_boilerplate_table(rows):
    joined = " ".join(cell for row in rows for cell in row)
    boilerplate_markers = [
        "Incorporated by Reference", "Exhibit Number", "/s/",
        "PCAOB Firm ID", "Power of Attorney", "Rule 10b5-1",
    ]
    return any(marker in joined for marker in boilerplate_markers)


# ---------------------------------------------------------------------------
# Chart entries
# ---------------------------------------------------------------------------

def find_perf_table(tables):
    """The 10-K's stock performance table (Apple vs S&P 500 vs Dow Jones Tech)."""
    for t in tables:
        rows = t.get("rows") or []
        if not rows:
            continue
        s = json.dumps(rows)
        if "Dow Jones" in s and "September" in " ".join(rows[0]):
            return t
    return None


def parse_perf_table(table):
    """-> (['September 2020', ...], {'Apple Inc.': ['100', '132', ...], ...})"""
    rows = table["rows"]
    header = [c.strip() for c in rows[0] if re.fullmatch(r"[A-Z][a-z]+ \d{4}", c.strip())]
    series = {}
    for r in rows[1:]:
        label = r[0].strip()
        vals = []
        for c in r[1:]:
            c = c.replace("$", "").replace(",", "").strip()
            if re.fullmatch(r"\d+(\.\d+)?", c):
                vals.append(c)
        if label and header and len(vals) == len(header):
            series[label] = vals
    return header, series


def chart_entries_for(filing_name, tables, meta, company_name):
    path = os.path.join(CHART_DIR, f"{filing_name}.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        charts = [c for c in json.load(f) if c.get("is_chart")]
    if not charts:
        return []
    table = find_perf_table(tables)
    if table is None:
        print(f"  [chart] {filing_name}: chart found but no performance table, skipping")
        return []
    header, series = parse_perf_table(table)
    if not series:
        print(f"  [chart] {filing_name}: could not parse performance table, skipping")
        return []

    title = charts[0].get("title") or "Comparison of 5-year cumulative total return"
    lines = [f"{name}: " + "; ".join(f"{h} ${v}" for h, v in zip(header, vals))
             for name, vals in series.items()]
    ranked = sorted(series.items(), key=lambda kv: -float(kv[1][-1]))
    summary = (f"At {header[-1]}: " + ", ".join(f"{n} ${v[-1]}" for n, v in ranked)
               + f". {ranked[0][0]} ended highest and {ranked[-1][0]} lowest.")
    text = (f"{company_name} stock performance graph, fiscal {meta['fiscal_year']} "
            f"{meta['form_type']} ({title}). Value of $100 invested in {header[0]}. "
            + " ".join(lines) + " " + summary)
    print(f"  [chart] {filing_name}: added chart entry ({len(series)} series, {len(header)} dates)")
    return [{
        "type": "chart",
        "embed_text": text,
        "display_text": text,
        "section": "Stock Performance Graph",
        **meta,
    }]


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading embedding model ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded.\n")

    files = {}
    for d in INPUT_DIRS:
        for p in glob.glob(os.path.join(d, "*.json")):
            files[os.path.basename(p)] = p
    print(f"Found {len(files)} classified filing(s) to index:")
    for name, p in sorted(files.items()):
        print(f"  {name}  <- {os.path.dirname(p)}/")
    print()

    all_entries = []
    for fname, classified_path in sorted(files.items()):
        with open(classified_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        filing_name = os.path.splitext(fname)[0]
        ticker, company_name, fiscal_year, form_type, fiscal_quarter = \
            company_info(filing_name)
        meta = {
            "filing": filing_name,
            "company": ticker,
            "fiscal_year": fiscal_year,
            "form_type": form_type,
            "fiscal_quarter": fiscal_quarter,
        }

        # text chunks
        for chunk in data["text"]:
            all_entries.append({
                "type": "text",
                "embed_text": chunk["text"],
                "display_text": chunk["text"],
                "section": chunk["section"],
                **meta,
            })

        # table rows (one entry per row)
        for table in data["tables"]:
            if is_boilerplate_table(table["rows"]):
                continue
            for row_chunk in table_to_row_chunks(table, filing_name, company_name):
                all_entries.append({
                    "type": "table",
                    "embed_text": row_chunk["text"],
                    "display_text": row_chunk["text"],
                    "section": f"Table {table['table_id']}",
                    **meta,
                })

        # chart entry (10-Ks with a stock performance graph)
        all_entries += chart_entries_for(filing_name, data["tables"], meta, company_name)

    print(f"\nEmbedding {len(all_entries)} total entries "
          f"(text + tables + charts) across all filings...")
    texts = [e["embed_text"] for e in all_entries]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    np.save(os.path.join(OUTPUT_DIR, "embeddings.npy"), embeddings)
    with open(os.path.join(OUTPUT_DIR, "entries.pkl"), "wb") as f:
        pickle.dump(all_entries, f)

    print(f"\nSaved {len(all_entries)} entries to {OUTPUT_DIR}/")
    for t in sorted({e['company'] for e in all_entries}):
        for form in ("10-K", "10-Q"):
            rows = [e for e in all_entries if e["company"] == t and e["form_type"] == form]
            if not rows:
                continue
            n_text = sum(1 for e in rows if e["type"] == "text")
            n_table = sum(1 for e in rows if e["type"] == "table")
            n_chart = sum(1 for e in rows if e["type"] == "chart")
            print(f"  {t} {form}: {len(rows)} entries "
                  f"({n_text} text, {n_table} table rows, {n_chart} chart)")