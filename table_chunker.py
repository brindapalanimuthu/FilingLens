import re

STATEMENT_HINTS = [
    ("Consolidated Balance Sheets", ["Total assets", "Total liabilities"]),
    ("Consolidated Statements of Operations", ["Total net sales", "Research and development", "Provision for income taxes"]),
    ("Consolidated Statements of Cash Flows", ["operating activities", "Depreciation and amortization"]),
    ("Consolidated Statements of Shareholders' Equity", ["Balance as of", "Common stock repurchased"]),
]

def clean_cells(row):
    return [c.strip() for c in row if c.strip() and c.strip() not in ("$", ")", "(")]

def guess_title(rows):
    labels = " ".join(c[0] for c in map(clean_cells, rows) if c)
    for title, keys in STATEMENT_HINTS:
        if any(k in labels for k in keys):
            return title
    return "Financial table"

def table_to_row_chunks(table, filing_name):
    rows = table["rows"]
    title = guess_title(rows)
    year_re = re.compile(r"\b(19|20)\d{2}\b")

    header = []
    for r in rows:
        cells = clean_cells(r)
        if cells and all(year_re.search(c) for c in cells):
            header = cells
            break

    chunks, section = [], ""
    for r in rows:
        cells = clean_cells(r)
        if not cells:
            continue
        if all(year_re.search(c) for c in cells):      # header row(s)
            continue
        if len(cells) == 1:                            # section header
            if cells[0].endswith(":"):
                section = cells[0].rstrip(":")
            continue
        label, vals = cells[0], cells[1:]
        full_label = label if label.lower().startswith("total") or not section \
            else f"{section} > {label}"
        if len(vals) == len(header):
            pairs = " | ".join(f"{h}: {v}" for h, v in zip(header, vals))
        else:
            pairs = " | ".join(vals)
        chunks.append({
            "type": "table_row",
            "text": f"{filing_name} | {title} | {full_label} | {pairs}",
            "table_id": table["table_id"],
        })
    return chunks