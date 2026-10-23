"""
FilingLens API — wraps the LangGraph app from filinglens_graph.py.

Run:
    pip install fastapi uvicorn
    export GEMINI_API_KEY=...        (PowerShell: $env:GEMINI_API_KEY="...")
    uvicorn api:app --reload
Then open http://127.0.0.1:8000
"""

import os
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from filinglens_graph import EMBED_MODEL_NAME, build_graph, load_index

state = {}  # filled once at startup, reused by every request


@asynccontextmanager
async def lifespan(app: FastAPI):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY before starting the server.")
    client = genai.Client(api_key=api_key)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings, entries = load_index()
    state["graph"] = build_graph(embed_model, embeddings, entries, client)
    state["n_entries"] = len(entries)
    yield
    state.clear()


app = FastAPI(title="FilingLens", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


_KV = re.compile(r"^(.{1,60}?):\s+(.+)$")


def parse_table_row(text):
    """Turn a row chunk like
    'FILING | Company | Financial table | Dilutive effect of awards | 2026: 24 | 2025: 32'
    into {'columns': ['2026', '2025'], 'rows': [{'label': ..., 'values': ['24', '32']}]}.
    Returns None if the text doesn't look like that, so the UI falls back to plain text."""
    parts = [p.strip() for p in text.split(" | ")]
    first_kv = next((i for i, p in enumerate(parts) if _KV.match(p)), None)
    if first_kv is None or first_kv < 1:
        return None
    label = parts[first_kv - 1]
    columns, values = [], []
    for p in parts[first_kv:]:
        m = _KV.match(p)
        if not m:
            return None
        columns.append(m.group(1))
        values.append(m.group(2))
    return {"columns": columns, "rows": [{"label": label, "values": values}]}


@app.get("/health")
def health():
    return {"status": "ok", "index_entries": state.get("n_entries", 0)}


# plain `def` (not async): the graph blocks on Gemini calls, so FastAPI
# runs it in a worker thread instead of freezing the event loop.
@app.post("/ask")
def ask(req: AskRequest):
    try:
        result = state["graph"].invoke({
            "query": req.question.strip(),
            "route": "",
            "retrieved": [],
            "answer": "",
            "verification": [],
            "model": "",
        })
    except RuntimeError as e:  # all Gemini models exhausted
        raise HTTPException(status_code=503, detail=str(e))

    sources = []
    for i, e in enumerate(result["retrieved"], 1):
        src = {
            "id": i,
            "company": e.get("company"),
            "fiscal_year": e.get("fiscal_year"),
            "filing": e["filing"],
            "section": e["section"],
            "type": e["type"],
            "text": e["display_text"][:1200],
        }
        # chart sources carry the path of the original chart image
        if e.get("image"):
            src["image_url"] = "/" + e["image"].lstrip("/")
        # real rows/columns for the UI: chart entries carry theirs, table rows are parsed
        table = e.get("table")
        if table is None and e["type"] == "table":
            table = parse_table_row(e["display_text"])
        if table:
            src["table"] = table
        sources.append(src)

    verification = result["verification"]
    return {
        "question": req.question,
        "answer": result["answer"],
        "route": result["route"],
        "model": result["model"],
        "sources": sources,
        "verification": verification,
        "unverified_count": sum(1 for v in verification if not v["verified"]),
    }


@app.get("/")
def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")

# downloaded chart images (created by fetch_chart_images.py)
if os.path.isdir("chart_images"):
    app.mount("/chart_images", StaticFiles(directory="chart_images"), name="chart_images")