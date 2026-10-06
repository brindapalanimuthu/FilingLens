"""
FilingLens API — wraps the LangGraph app from filinglens_graph.py.

Run:
    pip install fastapi uvicorn
    export GEMINI_API_KEY=...        (PowerShell: $env:GEMINI_API_KEY="...")
    uvicorn api:app --reload
Then open http://127.0.0.1:8000
"""

import os
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

    sources = [
        {
            "id": i,
            "company": e.get("company"),
            "fiscal_year": e.get("fiscal_year"),
            "filing": e["filing"],
            "section": e["section"],
            "type": e["type"],
            "text": e["display_text"][:1200],
        }
        for i, e in enumerate(result["retrieved"], 1)
    ]
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