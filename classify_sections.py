"""
FilingLens — Stage 3: Section Classification

Usage:
    python classify_sections.py MSFT             # classify only MSFT (default distilbert model)
    python classify_sections.py MSFT --deberta   # stronger model, writes to classified_deberta/
    python classify_sections.py --force          # redo files that already exist
"""

from transformers import pipeline
import json
import os
import glob
import sys


INPUT_DIR = "parsed"

USE_DEBERTA = "--deberta" in sys.argv
if USE_DEBERTA:
    MODEL_NAME = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
    OUTPUT_DIR = "classified_deberta"
else:
    MODEL_NAME = "typeform/distilbert-base-uncased-mnli"
    OUTPUT_DIR = "classified"

SECTION_LABELS = [
    "a description of the company's business, products, and services",
    "risks and uncertainties that could adversely affect the business",
    "ongoing lawsuits, legal proceedings, or government investigations",
    "physical facilities, offices, and real estate owned or leased by the company",
    "information about the company's stock price, dividends, and share repurchases",
    "management's analysis of financial results, revenue, and expenses",
    "exposure to market risk from interest rates, foreign currency, or financial instruments",
    "financial statements, balance sheets, income statements, and accounting notes",
    "internal controls over financial reporting and disclosure controls",
    "general administrative, legal, or procedural text such as signatures and certifications",
]

SECTION_NAMES = [
    "Business Overview",
    "Risk Factors",
    "Legal Proceedings",
    "Properties",
    "Market for Common Stock and Shareholder Matters",
    "Management's Discussion and Analysis",
    "Quantitative and Qualitative Disclosures About Market Risk",
    "Financial Statements and Notes",
    "Controls and Procedures",
    "Other",
]

CONFIDENCE_THRESHOLD = 0.3


def classify_chunks(text_chunks, classifier):
    label_to_name = dict(zip(SECTION_LABELS, SECTION_NAMES))
    results = []
    total = len(text_chunks)
    for i, text in enumerate(text_chunks):
        snippet = text[:800]
        prediction = classifier(snippet, SECTION_LABELS, multi_label=False)
        top_label = prediction["labels"][0]
        top_score = prediction["scores"][0]
        section = label_to_name[top_label] if top_score >= CONFIDENCE_THRESHOLD else "Other"
        results.append({
            "text": text,
            "section": section,
            "confidence": round(top_score, 3),
        })
        if (i + 1) % 25 == 0 or (i + 1) == total:
            print(f"    Classified {i + 1}/{total} chunks...")
    return results


def print_distribution(classified_text):
    counts = {}
    for c in classified_text:
        counts[c["section"]] = counts.get(c["section"], 0) + 1
    total = len(classified_text)
    for name, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {name:58} {n:4} ({n / total:.0%})")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    force = "--force" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ticker = args[0].upper() if args else None

    pattern = f"{ticker}_*.json" if ticker else "*.json"
    parsed_files = sorted(glob.glob(os.path.join(INPUT_DIR, pattern)))
    print(f"Model: {MODEL_NAME}")
    print(f"Output: {OUTPUT_DIR}/")
    print(f"Found {len(parsed_files)} parsed filing(s)")

    todo = []
    for p in parsed_files:
        out_path = os.path.join(OUTPUT_DIR, os.path.basename(p))
        if os.path.exists(out_path) and not force:
            print(f"Already classified {out_path}, skipping")
        else:
            todo.append(p)

    if not todo:
        print("Nothing to do.")
        sys.exit(0)

    print(f"\nLoading model... this may take a while on first run.")
    classifier = pipeline("zero-shot-classification", model=MODEL_NAME)
    print("Model loaded.\n")

    for parsed_path in todo:
        print(f"Classifying {parsed_path}...")
        with open(parsed_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        classified_text = classify_chunks(data["text"], classifier)

        output = {
            "tables": data["tables"],
            "images": data["images"],
            "text": classified_text,
        }

        out_path = os.path.join(OUTPUT_DIR, os.path.basename(parsed_path))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"  Saved: {out_path}")
        print("  Section distribution:")
        print_distribution(classified_text)
        print()