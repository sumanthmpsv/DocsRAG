# DocsRAG — Document Q&A API with source citations

Ask questions about a set of documents and get answers grounded in those
documents, with the exact source passages each answer was drawn from. Built as a
production-style retrieval-augmented generation (RAG) service: a FastAPI backend
you can integrate into any app, plus a simple UI for demos.

## What it does
1. **Ingests** PDF / TXT / Markdown files and splits them into overlapping chunks.
2. **Embeds** each chunk into a vector and stores it for similarity search.
3. **Answers** questions by retrieving the most relevant chunks, passing them to
   an LLM as context, and returning the answer plus its sources.

## Stack
FastAPI · OpenAI (embeddings + chat) · NumPy vector search · Streamlit · Python 3.10+

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then paste your OpenAI API key into .env
```

## Run

```bash
# 1. Build the index from the files in docs/ (a sample doc is included)
python3 build_index.py       # watch it chunk + embed

# 2a. Try it on the command line
python3 rag.py "What are the ways a batter can get out?"

# 2b. Or launch the demo UI
streamlit run app.py        # the visual demo

# 2c. Or run it as an API
uvicorn api:app --reload
#     then POST to http://127.0.0.1:8000/ask  (interactive docs at /docs)
```

Replace the file in `docs/` with your own documents and re-run `build_index.py`.

## How it works (architecture)
- `ingest.py` — load + chunk documents (no API calls)
- `store.py` — embed chunks, cosine similarity search (in-memory NumPy)
- `build_index.py` — build and persist the index to `index.pkl`
- `rag.py` — retrieve relevant chunks + generate a grounded, cited answer
- `api.py` — FastAPI `/ask` and `/health` endpoints
- `app.py` — Streamlit demo UI

## Limitations / next steps
- The index is in-memory and persisted to a pickle file; for production scale,
  swap the store for Postgres + pgvector (the search logic stays the same).
- No authentication on the API yet — add before any public deployment.
- Citations are the retrieved passages; per-sentence attribution is a possible
  enhancement.
- Answer quality depends on chunk size and `k`; both are easy to tune.
