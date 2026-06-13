"""
Milestone 4 — Expose the engine over HTTP with FastAPI.

This is what makes it a product a client can integrate, not just a script. The
index is loaded once at startup (not per request) so answers are fast.

Run:
    uvicorn api:app --reload

Then open http://127.0.0.1:8000/docs for an interactive test page, or:
    curl -X POST http://127.0.0.1:8000/ask \
         -H "Content-Type: application/json" \
         -d '{"question": "your question here"}'
"""
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import rag
from store import VectorStore

load_dotenv()

app = FastAPI(title="DocsRAG", description="Ask questions about your documents, with sources.")


class AskRequest(BaseModel):
    question: str
    k: int = 4


try:
    STORE = VectorStore.load("index.pkl")
except FileNotFoundError:
    STORE = None


@app.get("/health")
def health():
    return {"status": "ok", "indexed": STORE is not None}


@app.post("/ask")
def ask(req: AskRequest):
    if STORE is None:
        raise HTTPException(503, "Index not built. Run: python build_index.py")
    return rag.answer(req.question, k=req.k, store=STORE)
