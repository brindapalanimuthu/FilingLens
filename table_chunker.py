import re

STATEMENT_HINTS = [
    ("Consolidated Balance Sheets", ["Total assets", "Total liabilities"]),
    ("Consolidated Statements of Operations", [
        "Total net sales", "Research and development", "Provision for income taxes",
        "Total cost of revenue", "Gross margin",
    ]),
    ("Consolidated Statements of Cash Flows", [
        "operating activities", "Depreciation and amortization",
        "Net cash from operations", "Net cash used in",
    ]),
    ("Consolidated Statements of Shareholders' Equity", [
        "Balance as of", "Common stock repurchased", "Balance, beginning of period",
    ]),
]

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
YEAR_ONLY_RE = re.compile(r"(19|20)\d{2}")

# rows that end any open "Section:" group
SECTION_ENDERS = (
    "total", "net income", "operating income", "gross margin",
    "income before", "net cash", "net sales",
)


def clean_cells(row):
    return [c.strip() for c in row if c.strip() and c.strip() not in ("$", ")", "(")]


def guess_title(rows):
    labels = " ".join(c[0] for c in map(clean_cells, rows) if c)
    for title, keys in STATEMENT_HINTS:
        if any(k in labels for k in keys):
            return title
    return "Financial table"


def year_cells(cells):
    return [c for c in cells if YEAR_ONLY_RE.fullmatch(c)]


def is_header_row(cells):
    if all(YEAR_RE.search(c) for c in cells):
        return True
    yrs = year_cells(cells)
    return len(yrs) >= 2 and len(yrs) >= len(cells) - 1


def is_plain_section_title(text):
    """Single-cell row without a colon that looks like a group title,
    e.g. 'Intelligent Cloud' or 'Total' (no digits, short, not a unit note)."""
    return (
        len(text) <= 60
        and not any(ch.isdigit() for ch in text)
        and not text.startswith("(")
    )


def table_to_row_chunks(table, filing_name, company_name=""):
    rows = table["rows"]
    title = guess_title(rows)
    prefix = f"{filing_name} | {company_name}" if company_name else filing_name
    is_generic_table = title == "Financial table"

    header = []
    for r in rows:
        cells = clean_cells(r)
        if not cells:
            continue
        if all(YEAR_RE.search(c) for c in cells):          # Apple-style header
            header = cells
            break
        yrs = year_cells(cells)
        if len(yrs) >= 2 and len(yrs) >= len(cells) - 1:   # e.g. "Year Ended June 30," 2025 2024 2023
            header = yrs
            break

    chunks, section = [], ""
    for r in rows:
        cells = clean_cells(r)
        if not cells:
            continue
        if is_header_row(cells):
            continue
        if len(cells) == 1:                                # section header
            if cells[0].endswith(":"):
                section = cells[0].rstrip(":")
            elif is_generic_table and is_plain_section_title(cells[0]):
                section = cells[0]                         # e.g. segment name, or "Total"
            continue
        label, vals = cells[0], cells[1:]
        if section.lower() == "total":
            full_label = f"Total {label[0].lower()}{label[1:]}"   # 'Total operating income'
        elif label.lower().startswith("total") or not section:
            full_label = label
        else:
            full_label = f"{section} > {label}"
        if len(vals) == len(header):
            pairs = " | ".join(f"{h}: {v}" for h, v in zip(header, vals))
        else:
            pairs = " | ".join(vals)
        chunks.append({
            "type": "table_row",
            "text": f"{prefix} | {title} | {full_label} | {pairs}",
            "table_id": table["table_id"],
        })
        if label.lower().startswith(SECTION_ENDERS) and section.lower() != "total":
            section = ""
    return chunks