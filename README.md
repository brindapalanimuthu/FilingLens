# 🔎 FilingLens

**Ask analyst-style questions over SEC filings. Get cited answers where every number is checked against the source.**

FilingLens is an agentic RAG system over SEC 10-K and 10-Q filings. It understands prose, financial tables *and* the stock performance chart, routes each question to the right retrieval strategy, and verifies the numbers in the generated answer against the retrieved evidence before showing it. If the filings don't contain the answer, it says so instead of guessing.

```
Question → Route → Retrieve evidence → Generate → Verify numbers → Cited answer
```

> **Scope today:** Apple and Microsoft 10-K (3 fiscal years each) and 10-Q (6 quarters each) filings. Text + table retrieval, quarter-aware, plus stock performance chart retrieval for Apple's 10-Ks. 8-K support and general visual retrieval are roadmap items, not built.


---

## Why it exists

Plain RAG over filings fails in predictable ways:

1. **Table rows get crowded out.** Fluent prose chunks outscore the one financial-statement row a question needs.
2. **LLMs misquote numbers.** A confident answer with a wrong figure is worse than no answer.
3. **Vision models misread charts.** Reading values off a line chart gives plausible but wrong numbers.

FilingLens addresses these with row-level table chunking, hybrid retrieval, a post-generation numeric verification step, and by sourcing chart values from the exact table printed under each graph.

## Architecture

```mermaid
flowchart LR
    A[SEC EDGAR] --> B[Parse<br/>text / tables / images]
    B --> C[Section tagging<br/>zero-shot DeBERTa]
    B --> I[Chart detection<br/>Gemini vision]
    C --> D[(Index<br/>text chunks, table rows, chart entries<br/>company, fiscal year, section)]
    I --> D
    Q[Question] --> R{Route<br/>LangGraph}
    R -->|numeric| T[Hybrid table search<br/>BM25 + embeddings + label match]
    R -->|explanatory| X[Text search<br/>embeddings + section boost]
    R -->|chart| H[Chart search<br/>fiscal-year matching]
    D --> T
    D --> X
    D --> H
    T --> G[Generate<br/>Gemini, model fallback]
    X --> G
    H --> G
    G --> V[Verify numbers<br/>against retrieved source]
    V --> O[Answer + citations<br/>+ number-check chips<br/>+ source tables and chart images]
```

| Stage | What it does | File |
|---|---|---|
| Ingest | Downloads filings from SEC EDGAR by ticker and form type | `edgar_ingest.py` |
| Parse | Splits filings into tables, images and ≤1500-char text chunks | `parse_filing.py` |
| Classify | Tags text chunks with their 10-K section (zero-shot DeBERTa-v3) | `classify_sections.py` |
| Charts | Downloads chart images from EDGAR; Gemini vision flags real charts vs logos | `fetch_chart_images.py`, `describe_charts.py` |
| Table chunking | One index entry per table row, with label, period and section | `table_chunker.py` |
| Index | Embeds text, table rows and chart entries (MiniLM), tagged by company, form type and fiscal year/quarter | `build_full_index.py` |
| Retrieve | Hybrid search, company filter, segment-aware re-ranking | `hybrid_search.py` |
| Orchestrate | LangGraph: classify query → retrieve → generate → verify | `filinglens_graph.py` |
| Serve | FastAPI (`/ask`, `/health`) + web UI | `api.py`, `static/`, `frontend/` |
| Evaluate | Eval harness (main, held-out, 10-Q sets) and retrieval ablation | `eval_harness.py`, `ablation_retrieval.py` |

## Key features

**Agentic query routing.** A LangGraph node classifies each question. Numeric questions ("What was Apple's net income in 2024?") go to table-focused retrieval; explanatory ones ("Why did margins change?") go to text-focused retrieval; questions about the stock performance graph go to chart retrieval.

**Table-aware retrieval.** Every table row becomes its own searchable unit (label, period, section), so questions about revenue, EPS, operating income, segments or year-over-year changes retrieve the exact row.

**Hybrid search.** Table retrieval fuses three signals with reciprocal rank fusion: embedding similarity, BM25 keyword match, and IDF-weighted financial-label match. It expands aliases (R&D → Research and Development, EPS → Earnings Per Share), splits multi-part questions, and demotes percentage rows when the question wants dollar amounts.

**Company and fiscal-year awareness.** The retriever detects the company in the question (Apple/AAPL/iPhone, Microsoft/MSFT/Azure…), filters to it, and prefers the filing for the requested year. It also applies a segment-aware re-rank so "Services revenue" answers from the Services row.

**Chart retrieval with exact numbers.** Apple's 10-Ks contain a 5-year cumulative total return graph (Apple vs the S&P 500 vs the Dow Jones U.S. Technology index). FilingLens downloads each chart image and uses Gemini vision only to confirm it is a chart (logos are discarded). Vision models misread values off line charts, so the indexed numbers come from the exact table printed beneath each graph, parsed from the filing itself. The original image is shown next to the answer in the UI.

Each 10-K restarts its graph at $100 from its own base year, so the same calendar date has different values in different filings (Apple at September 2024 is $208 in the FY2025 filing and $430 in the FY2024 filing). The chart route matches the fiscal year named in the question.

**Source-cited answers.** Answers cite `[Source N]`; the UI shows the retrieved sources as real tables (line item × period) and, for charts, the original image, so the evidence can be inspected.

**Numeric verification.** Numbers in the answer are extracted, normalized ($ , % parentheses), and matched against the retrieved evidence. Each is marked verified or not.

```
✓ $391,035 million     ✓ 6.8%     ✗ $405,200 million   ← flagged
```

**Quarter-aware retrieval (10-Q).** Fiscal year and quarter are derived per filing (a Microsoft 10-Q filed Oct 2024 is Q1 FY2025). Questions that mention a quarter search 10-Q rows and boost the quarter named; everything else stays on 10-K so annual figures aren't polluted by quarterly rows. Apple's multi-column 10-Q tables keep their "Three / Six Months Ended" labels so quarter and year-to-date figures aren't confused.

**Refuses when unsupported.** Questions the filings can't answer (a company not in the index, a future year, or a chart that isn't in the filings) are part of the eval. The system says it doesn't know.


**Model fallback.** If one Gemini model hits quota, the system switches to the next configured model.

## Results

| Eval set | Questions | Correct | Retrieval hit | Numeric verification |
|---|---|---|---|---|
| Main (numeric, derived, prose, chart, unanswerable) | 39 | **39 / 39** | 33 / 33 | 98% |
| Held-out, 10-K (written after tuning: segments, unanswerable) | 11 | **11 / 11** (baseline 10 / 11) | 9 / 9 | 100% |
| 10-Q (single quarter, two quarters, quarter-over-quarter, unanswerable) | 9 | **9 / 9** | 8 / 8 | 88% |

The main set includes 7 chart questions (exact values from the FY2023, FY2024 and FY2025 graphs, a multi-series question, and a Microsoft question that must be refused because the Microsoft 10-Ks in the index contain no stock performance graph). All 7 passed on their first run.

Numeric verification excludes dates and bare years. It is lower on derived questions (e.g. "by how much did revenue increase") because the computed difference isn't itself in any source; only its inputs are, and those are verified.

### Retrieval ablation

Does each retrieval component earn its place? For every answerable eval question, `ablation_retrieval.py` checks whether the numbers needed to answer it appear in the top-6 table rows sent to the LLM. It makes no LLM calls. All methods share the same company/form-type filter, candidate pool and fiscal-year picking, so only the ranking differs. (It covers table retrieval only; chart questions are not part of the ablation.)

| Retrieval method | Main (10-K) | Held-out (10-K) | 10-Q | Overall |
|---|---|---|---|---|
| Embeddings only | 14 / 24 | 2 / 9 | 3 / 8 | 19 / 41 (46%) |
| BM25 only | 17 / 24 | 9 / 9 | 2 / 8 | 28 / 41 (68%) |
| **Hybrid (FilingLens)** | **24 / 24** | **9 / 9** | **8 / 8** | **41 / 41 (100%)** |

Embeddings alone often retrieve the right *kind* of row from the wrong company, year or segment. BM25 alone matches line-item names well but fails when wording differs or when the same label appears in several periods and filings (e.g. Microsoft net income across fiscal years, or 10-Q quarter columns). The hybrid ranker, with label matching and segment/quarter-aware re-ranking, is the only method that gets everything.

**How to read this honestly:** the hybrid ranker was tuned against the main and held-out sets, and the baselines were not tuned at all, so 41/41 overstates the gap on unseen questions. The 10-Q set is the cleanest comparison: its questions were written after the retrieval code was finished and the hybrid ranker passed 8/8 on the first run with no tuning against them.

Early development: row-level table chunking plus hybrid search took a 12-question eval from 6/12 to 12/12.

## Design decisions

- **Row-level table chunks.** Whole-table embeddings buried the one row a question needs.
- **Hybrid, not embeddings-only.** Embeddings miss exact line-item names; BM25 and label matching catch them.
- **Type-stratified retrieval.** Table slots are guaranteed so prose can't crowd them out.
- **Verify, don't trust.** The LLM never gets the last word on a number.
- **Chart values come from the filing's table, not the vision model.** In testing, Gemini got the endpoints of the Apple chart right but misread interior points (it placed Apple's 2022 dip wrongly and mixed up two series), so vision is used only to detect charts.
- **Separate retrieval from generation.** The LLM only sees retrieved evidence; it never searches the filing itself.

## Limitations

- Two companies: three fiscal years of 10-Ks and six quarters of 10-Qs each. Results don't claim to generalize across sectors or filers.
- Chart retrieval covers the stock performance graph in Apple's 10-Ks only. The Microsoft 10-Ks in the index contain no stock performance graph, and the other images in the filings (logos) are discarded. There is no general visual retrieval (e.g. ColPali); that is future work.
- Microsoft 10-Q tables with year-to-date columns (Q2, Q3) are labeled by year only, so three-month and year-to-date figures aren't explicitly distinguished in the index. The 10-Q eval passes, but this is a known weakness.
- Eval sets are small and written by the author. 100% is on *these* sets, not a public benchmark. The held-out set was used to find and fix failures (baseline 10/11 → 11/11), so it is no longer a pure test set; the 10-Q set and the chart questions, which passed on their first run, are the cleanest evidence.
- Query routing is keyword-heuristic, not learned.
- Verification checks that a number appears in the retrieved evidence. It is not a proof that the reasoning is correct. Calculated figures (growth rates, differences) can't appear verbatim in a filing, so they are expected to show as unverified while their inputs are verified.
- No 8-K support.
- Generation uses the Gemini free tier and depends on its quotas.
- The index is local and must be rebuilt when filings are added.

## Quickstart

```bash
git clone https://github.com/brindapalanimuthu/FilingLens.git && cd FilingLens
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY="your_key"          # Google AI Studio free tier

# Build the data (generated files are not committed)
python edgar_ingest.py AAPL 10-K
python edgar_ingest.py MSFT 10-K
python edgar_ingest.py AAPL 10-Q          # optional: quarterly filings
python edgar_ingest.py MSFT 10-Q
python parse_filing.py
python classify_sections.py

# Optional: chart retrieval (Apple 10-Ks). Set your own name and email in the
# User-Agent inside fetch_chart_images.py first, as SEC EDGAR requires.
python fetch_chart_images.py
python describe_charts.py

python build_full_index.py

# Run the API + UI
uvicorn api:app --reload                  # http://127.0.0.1:8000

# Optional: React frontend
cd frontend && npm install && npm run dev # http://127.0.0.1:5173

# Evaluate
python ablation_retrieval.py              # retrieval-only comparison, no LLM calls
python eval_harness.py
EVAL_SET=eval_holdout.json EVAL_RESULTS=holdout_results.json python eval_harness.py --fresh
EVAL_SET=eval_10q.json EVAL_RESULTS=eval_10q_results.json python eval_harness.py --fresh
```

## API

`POST /ask`

```json
{ "question": "What was Apple's Services revenue in 2024?" }
```

```json
{
  "answer": "...",
  "route": "table_focused",
  "model": "gemini-2.5-flash",
  "sources": [
    {
      "id": 1,
      "company": "AAPL",
      "fiscal_year": 2024,
      "section": "Table 5",
      "type": "table",
      "text": "...",
      "table": { "columns": ["2024", "2023"], "rows": [{ "label": "Services", "values": ["96,169", "85,200"] }] }
    }
  ],
  "verification": [{ "number": "$391,035", "verified": true }],
  "unverified_count": 0
}
```

Chart sources (`"type": "chart"`) also include an `image_url` and a `table` with one row per series and one column per date. `GET /health` → `{ "status": "ok", "index_entries": ... }`

## Example questions

- *What was Apple's total net sales in 2024?* (table route)
- *How did Microsoft's revenue change between 2023 and 2024?* (multi-year, derived)
- *What were Apple's R&D expenses and net income in 2024?* (multi-part)
- *Why did Apple's gross margin change?* (text route)
- *How did Apple's stock perform against the S&P 500 over the five years to 2025?* (chart route)
- *What was Alphabet's revenue?* (not in index → refuses)

## Roadmap

- [x] 10-K ingestion, parsing, section tagging, hybrid retrieval, verification, eval
- [x] FastAPI + web UI
- [x] 10-Q indexing, fiscal-quarter detection, quarter-over-quarter questions
- [x] Stock performance chart retrieval (Apple 10-Ks), exact values from the filing's table
- [ ] Explicit three-month vs year-to-date labels for Microsoft 10-Q tables
- [ ] Learned query router
- [ ] General visual retrieval (ColPali / ColQwen) for charts and figures beyond the performance graph
- [ ] 8-K support, more companies, persistent vector store

## Stack

Python · LangGraph · sentence-transformers (all-MiniLM-L6-v2) · rank_bm25 · Hugging Face zero-shot (DeBERTa-v3) · Gemini (text + vision) · FastAPI · React + Vite · SEC EDGAR

## Author

**Brinda Palanimuthu** · [GitHub](https://github.com/brindapalanimuthu)

Built for educational, research and portfolio purposes. Data from SEC EDGAR.