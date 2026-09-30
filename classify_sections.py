"""
FilingLens — Stage 3: Section Classification
"""

from transformers import pipeline
import json
import os
import glob


INPUT_DIR = "parsed"
OUTPUT_DIR = "classified"

MODEL_NAME = "typeform/distilbert-base-uncased-mnli"

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


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading model ({MODEL_NAME})... this may take a while on first run.")
    classifier = pipeline("zero-shot-classification", model=MODEL_NAME)
    print("Model loaded.\n")

    parsed_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    print(f"Found {len(parsed_files)} parsed filing(s) to classify")

    for parsed_path in parsed_files:
        print(f"\nClassifying {parsed_path}...")
        with open(parsed_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        text_chunks = data["text"]
        classified_text = classify_chunks(text_chunks, classifier)

        output = {
            "tables": data["tables"],
            "images": data["images"],
            "text": classified_text,
        }

        basename = os.path.splitext(os.path.basename(parsed_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{basename}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"  Saved: {out_path}")
