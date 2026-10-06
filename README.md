# 🔎 FilingLens

### Agentic RAG for Analyst-Grade Financial Research

**FilingLens** is an AI-powered financial research assistant that lets users ask natural-language questions about SEC company filings and receive **source-grounded answers with citations and numeric verification**.

Instead of treating a financial filing as plain text, FilingLens understands both **prose and financial tables**, routes questions to the appropriate retrieval strategy, and verifies numerical claims against the retrieved evidence.

> **Ask a question → Retrieve evidence → Generate an answer → Verify the numbers**

---

## ✨ Why FilingLens?

SEC filings such as 10-Ks contain hundreds of pages of:

* Financial statements
* Tables
* Risk disclosures
* Management discussion
* Business information
* Segment-level metrics
* Year-over-year financial data

Finding a specific answer manually can be time-consuming.

FilingLens turns this process into a conversational research workflow.

### Example

Instead of searching through hundreds of pages for:

> **"What was Apple's Services revenue in 2024 and how did it change from 2023?"**

FilingLens retrieves the relevant financial table rows, identifies the appropriate filing year, generates the answer, cites the evidence, and checks whether the numerical claims actually appear in the retrieved sources.

---

## 🚀 Key Features

### 🧠 Agentic Query Routing

FilingLens classifies incoming questions and dynamically chooses between:

* **Text-focused retrieval** for explanatory and qualitative questions
* **Table-focused retrieval** for financial and numerical questions

Examples:

```text
"What risks does Apple identify?"
        ↓
Text-focused retrieval
```

```text
"What was Apple's net income in 2024?"
        ↓
Table-focused retrieval
```

---

### 📊 Table-Aware Financial Retrieval

Financial tables are not treated as ordinary text.

FilingLens converts table rows into searchable retrieval units, allowing questions about:

* Revenue
* Net income
* Operating income
* EPS
* Expenses
* Assets
* Debt
* Margins
* Segment revenue
* Year-over-year changes

to retrieve the relevant financial rows.

---

### 🔍 Hybrid Search

Table retrieval combines multiple signals:

```text
User Query
    │
    ├── Semantic Search
    │      └── Sentence Transformers
    │
    ├── BM25 Keyword Search
    │
    └── Label Matching
           └── Financial row labels
                    │
                    ▼
             Reciprocal Rank Fusion
                    │
                    ▼
             Re-ranked Results
```

This helps handle both semantic queries and exact financial terminology.

The implementation also supports query expansion for terms such as:

```text
R&D      → Research and Development
EPS      → Earnings Per Share
SG&A     → Selling, General and Administrative
CapEx    → Additions to Property and Equipment
```

---

### 🏢 Multi-Company Retrieval

Queries can automatically detect the company being discussed.

Currently supported company aliases include:

* Apple / AAPL / iPhone / iPad
* Microsoft / MSFT / Azure / Windows / Xbox / LinkedIn

The retrieval layer can then restrict results to the relevant company.

---

### 📅 Fiscal-Year-Aware Retrieval

When a question specifies a year, FilingLens prioritizes evidence from the corresponding filing.

For example:

```text
"What was Microsoft's revenue in 2024?"
```

The retrieval process gives priority to the relevant fiscal-year filing rather than mixing figures from unrelated filings.

---

### 📑 Source-Cited Answers

Generated answers reference the retrieved evidence:

```text
Apple's Services revenue increased in 2024...
[Source 2]
```

The UI also exposes the underlying retrieved sources so users can inspect the evidence themselves.

---

### ✅ Numeric Claim Verification

Financial AI systems can be especially dangerous when they produce incorrect numbers.

FilingLens therefore performs an additional verification step after generation.

```text
Generated Answer
       ↓
Extract numerical claims
       ↓
Normalize values
       ↓
Compare against retrieved sources
       ↓
Verified / Not Verified
```

Example:

```text
✓ $391,035 million
✓ 6.8%
✗ $405,200 million
```

This provides an additional layer of protection against unsupported numerical claims.

---

### 🔄 Gemini Model Fallback

FilingLens includes model fallback logic.

If one Gemini model becomes unavailable or reaches its quota, the system can switch to another configured model instead of immediately failing the request.

---

## 🏗️ Architecture

```text
                    SEC EDGAR
                       │
                       ▼
              Filing Ingestion
                       │
                       ▼
                HTML Filings
                       │
                       ▼
              Document Parsing
                ┌──────┼──────┐
                ▼      ▼      ▼
              Text   Tables  Images
                │      │
                ▼      ▼
          Text Chunks  Table Rows
                │      │
                └──┬───┘
                   ▼
          Section Classification
                   │
                   ▼
        Sentence Transformer Embeddings
                   │
                   ▼
              Full Index
          ┌────────┴────────┐
          │                 │
      Text Index        Table Index
          │                 │
          └────────┬────────┘
                   ▼
              User Question
                   │
                   ▼
          LangGraph Query Router
             ┌─────┴─────┐
             ▼           ▼
        Text Route    Table Route
             │           │
             │      Hybrid Search
             │      ┌────┼────┐
             │      ▼    ▼    ▼
             │    BM25 Embedding
             │         Label Match
             │           │
             └─────┬─────┘
                   ▼
             Retrieved Evidence
                   │
                   ▼
              Gemini LLM
                   │
                   ▼
             Grounded Answer
                   │
                   ▼
           Numeric Verification
                   │
                   ▼
            FastAPI Response
                   │
                   ▼
             Web Interface
```

---

## 🧩 Tech Stack

| Component            | Technology              |
| -------------------- | ----------------------- |
| Language             | Python                  |
| LLM                  | Google Gemini           |
| Agent Orchestration  | LangGraph               |
| Embeddings           | Sentence Transformers   |
| Semantic Model       | `all-MiniLM-L6-v2`      |
| Keyword Retrieval    | BM25                    |
| Numerical Processing | NumPy                   |
| Document Parsing     | BeautifulSoup           |
| Financial Data       | SEC EDGAR               |
| API                  | FastAPI                 |
| Frontend             | HTML / CSS / JavaScript |
| Retrieval            | Hybrid Semantic + BM25  |
| Data Storage         | NumPy + Pickle index    |

---

## 📂 Project Structure

```text
FilingLens/
│
├── static/
│   └── index.html
│
├── classified_deberta/
│
├── full_index/
│   ├── embeddings.npy
│   └── entries.pkl
│
├── filings_raw/
│
├── parsed/
│
├── api.py
├── edgar_ingest.py
├── parse_filing.py
├── classify_sections.py
├── table_chunker.py
├── build_index.py
├── build_full_index.py
├── hybrid_search.py
├── filinglens_graph.py
│
├── generate_answer.py
├── generate_answer_v2.py
│
├── eval_harness.py
├── eval_set.json
├── eval_holdout.json
│
├── requirements.txt
└── README.md
```

---

# ⚙️ How It Works

## 1. Ingest SEC Filings

FilingLens retrieves filings directly from the **SEC EDGAR system** using a company ticker.

Example:

```bash
python edgar_ingest.py AAPL
```

The ingestion pipeline resolves the company's CIK, retrieves recent filings, and downloads the filing documents.

---

## 2. Parse the Filing

The downloaded HTML filing is processed into separate components:

```text
SEC Filing
    │
    ├── Text
    ├── Tables
    └── Images
```

Text is split into manageable chunks while tables are preserved separately.

```bash
python parse_filing.py
```

---

## 3. Classify Sections

Text chunks are associated with filing sections such as:

```text
Risk Factors
Business
Properties
Legal Proceedings
Financial Statements
```

This gives the retrieval system additional context when answering questions.

---

## 4. Build the Search Index

FilingLens converts text and table rows into embeddings using:

```text
all-MiniLM-L6-v2
```

The resulting index contains:

```text
Text chunks
+
Table rows
+
Company
+
Fiscal year
+
Filing
+
Section
```

Build the combined index with:

```bash
python build_full_index.py
```

---

## 5. Route the Question

The LangGraph workflow first determines what type of question was asked.

```text
Question
   │
   ├── Numerical?
   │      └── Table-focused
   │
   └── Explanatory?
          └── Text-focused
```

This prevents purely semantic retrieval from being the only strategy for numerical financial questions.

---

## 6. Retrieve Evidence

For table-oriented questions, FilingLens uses:

```text
Embedding similarity
        +
BM25 keyword matching
        +
Financial label matching
        ↓
Reciprocal Rank Fusion
        ↓
Re-ranking
```

For text-oriented questions, semantic retrieval is combined with section-aware ranking.

---

## 7. Generate the Answer

The retrieved evidence is passed to Gemini with instructions to:

* Use only retrieved sources
* Cite sources
* Preserve financial numbers exactly
* Prefer the correct fiscal year
* Respect the requested company
* Use the requested segment or line item
* Explicitly state when evidence is insufficient
* Avoid guessing

---

## 8. Verify Numerical Claims

After generation, FilingLens extracts numerical claims from the response and checks whether those values appear in the retrieved evidence.

```text
Answer
  ↓
Find numbers
  ↓
Normalize numbers
  ↓
Search retrieved evidence
  ↓
Verification result
```

The API exposes:

```json
{
  "number": "$391,035",
  "verified": true
}
```

---

# 🖥️ Running FilingLens

## 1. Clone the Repository

```bash
git clone https://github.com/brindapalanimuthu/FilingLens.git
cd FilingLens
```

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The application also requires the API/runtime dependencies used by the FastAPI server and hybrid table retrieval:

```bash
pip install fastapi uvicorn rank-bm25 lxml
```

---

## 4. Configure Gemini

Set your Gemini API key.

### macOS / Linux

```bash
export GEMINI_API_KEY="your_api_key"
```

### Windows PowerShell

```powershell
$env:GEMINI_API_KEY="your_api_key"
```

---

## 5. Prepare the Index

If building the dataset from scratch:

```bash
python edgar_ingest.py AAPL
```

Then:

```bash
python parse_filing.py
```

Run the classification/indexing pipeline and finally:

```bash
python build_full_index.py
```

The generated index contains the embeddings and searchable filing entries.

---

## 6. Start the API

```bash
uvicorn api:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

You can also check the API health:

```text
GET /health
```

---

# 🔌 API

## Ask a Question

### Endpoint

```http
POST /ask
```

### Request

```json
{
  "question": "What was Apple's Services revenue in 2024?"
}
```

### Response

```json
{
  "question": "What was Apple's Services revenue in 2024?",
  "answer": "...",
  "route": "table_focused",
  "model": "gemini-2.5-flash",
  "sources": [],
  "verification": [],
  "unverified_count": 0
}
```

---

## Health Check

```http
GET /health
```

Example:

```json
{
  "status": "ok",
  "index_entries": 12500
}
```

---

# 🧪 Evaluation

FilingLens includes an evaluation workflow for measuring retrieval and answer quality.

Relevant files include:

```text
eval_set.json
eval_holdout.json
eval_results.json
eval_harness.py
```

This allows the system to be evaluated on predefined financial questions rather than relying only on manual testing.

---

# 🎯 Example Questions

FilingLens can handle questions such as:

### Financial Metrics

```text
What was Apple's total revenue in 2024?
```

### Comparisons

```text
How did Microsoft's revenue change between 2023 and 2024?
```

### Segment Analysis

```text
What was Apple's Services revenue in 2024?
```

### Explanatory Questions

```text
What risks does Apple identify in its latest filing?
```

### Financial Changes

```text
Why did operating income decrease?
```

### Multiple Metrics

```text
What were Apple's R&D expenses and net income in 2024?
```

---

# 🛡️ Design Principles

FilingLens is built around several principles for reliable financial question answering.

### 1. Ground answers in evidence

The model is instructed to answer only from retrieved sources.

### 2. Preserve financial numbers

Retrieved table values should be reproduced accurately rather than casually rounded.

### 3. Separate retrieval from generation

The LLM does not directly search the entire filing. Evidence is retrieved first.

### 4. Verify numerical claims

Generated numbers are checked against the retrieved evidence.

### 5. Prefer uncertainty over hallucination

If the available sources are insufficient, the system is instructed to say so rather than inventing an answer.

---

# 🔬 What Makes FilingLens Different?

Traditional RAG systems often follow:

```text
PDF
 ↓
Text chunks
 ↓
Vector database
 ↓
LLM
```

FilingLens is designed specifically around the structure of **financial filings**:

```text
SEC Filing
    │
    ├── Prose
    │
    ├── Financial Tables
    │
    ├── Sections
    │
    ├── Companies
    │
    └── Fiscal Years
           │
           ▼
     Query-Aware Routing
           │
     ┌─────┴─────┐
     ▼           ▼
Text Search   Table Search
                  │
          BM25 + Embeddings
                  │
                  ▼
          Financial Evidence
                  │
                  ▼
             LLM Answer
                  │
                  ▼
          Numeric Verification
```

The goal is not simply to build a chatbot over documents, but to build a **financial research system that understands the difference between narrative evidence and structured financial data**.

---

# 🚧 Current Limitations

FilingLens is an evolving research/portfolio project.

Current limitations include:

* SEC ingestion is currently centered on 10-K filings.
* Company-specific retrieval aliases are currently configured for a limited set of companies.
* Numeric verification checks whether values occur in retrieved evidence; it is not a complete mathematical proof of every generated claim.
* The system depends on the availability and quotas of the configured Gemini models.
* The local index must be rebuilt when new filings are added.

---

# 🗺️ Future Improvements

Potential next steps include:

* [ ] Support automated 10-Q and 8-K ingestion
* [ ] Expand company coverage
* [ ] Add persistent vector database storage
* [ ] Add financial-chart understanding
* [ ] Improve table structure preservation
* [ ] Add cross-company financial comparison
* [ ] Add temporal financial trend analysis
* [ ] Add confidence scoring
* [ ] Improve claim-level verification
* [ ] Add automated evaluation dashboards
* [ ] Add SEC filing update automation
* [ ] Add richer analyst reports and summaries

---

# 💡 Project Highlights

FilingLens demonstrates practical implementation of:

* **Agentic RAG**
* **LangGraph orchestration**
* **Hybrid retrieval**
* **Semantic search**
* **BM25 search**
* **Table-aware retrieval**
* **Financial document parsing**
* **LLM grounding**
* **Source attribution**
* **Numerical claim verification**
* **FastAPI deployment**
* **Evaluation-driven development**

---

# 📚 Data Source

Financial filings are retrieved from the **U.S. Securities and Exchange Commission (SEC) EDGAR system**.

SEC EDGAR:

https://www.sec.gov/edgar

---

# 👩‍💻 Author

**Brinda Palanimuthu**

Machine Learning Engineer | Applied AI | RAG | LLM Systems

GitHub:
https://github.com/brindapalanimuthu

---

# 📄 License

This project is intended for educational, research, and portfolio purposes.

See the repository license for applicable terms.
