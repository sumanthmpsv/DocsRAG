# DocsRAG — Deployable embeddable document assistant

An embeddable Q&A assistant that answers from a client's own documents, with
source citations, running on infrastructure the client controls. Backed by
Postgres + pgvector, API-key authenticated, and embeddable on any website with
a single `<script>` tag.

## What makes this more than a chatbot
- **Runs on the client's data**, behind a key they control — not a public tool.
- **Cites its sources** for every answer, and says "I don't know" when the
  documents don't cover the question.
- **Surfaces conflicting information** instead of silently picking one version.
- **Embeds anywhere** with one line, on the client's own domain.

## Architecture
- `ingest.py` — load + clean (strip page boilerplate) + chunk documents
- `pg_store.py` — Postgres + pgvector storage and cosine-similarity search
- `pg_build_index.py` — embed and load the index into Postgres
- `pg_rag.py` — retrieve + generate a grounded, cited, conflict-aware answer
- `embed_api.py` — FastAPI: `/ask` (auth + rate limit), `/widget.js`, demo `/`
- `schema.sql` — database table + HNSW index
- `render.yaml` — one-click deploy blueprint

## Run locally
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill in OPENAI_API_KEY, DATABASE_URL, DOCSRAG_API_KEY

# one-time DB setup (needs a local or hosted Postgres)
psql "$DATABASE_URL" -f schema.sql

# build the index, then run
python pg_build_index.py
uvicorn embed_api:app --reload
# open http://127.0.0.1:8000/  -> the mock client site with the Ask widget
```

## Deploy to Render
1. Push this folder to a GitHub repo.
2. Render > New > Blueprint > select the repo (uses `render.yaml`).
3. In the dashboard set `OPENAI_API_KEY` and `ALLOWED_ORIGINS`.
4. In the Render shell: `psql "$DATABASE_URL" -f schema.sql` then `python pg_build_index.py`.
5. Your live URL serves the demo page and the embeddable widget.

## How a client embeds it
```html
<script src="https://your-deploy-url/widget.js"
        data-docsrag data-key="THEIR_API_KEY"></script>
```

## Security notes
- `/ask` requires the `X-API-Key` header; set `DOCSRAG_API_KEY` to a long random string.
- CORS is locked to `ALLOWED_ORIGINS` — set it to the client's domain(s), never `*`.
- In-memory rate limiting guards cost; for multi-instance deploys move it to Redis.

## Known limits / next work (honest)
- Boilerplate stripping uses document-specific patterns; production needs a
  general repeated-line detector.
- Large/diverse corpora benefit from a re-ranking step after retrieval.
- Table-heavy documents may need dedicated table extraction.

---

# Loom walkthrough script (~2 minutes)

Record screen + voice. Keep it tight; this is a portfolio asset and a sales demo.

**[0:00–0:20] The problem.**
"Businesses have a pile of documents — manuals, policies, FAQs — and people can't
find answers in them quickly. General tools like ChatGPT or NotebookLM can't run
inside a company's own product or on data they need to keep private. So I build
an assistant that does."

**[0:20–0:50] Show it working.**
Open the live demo page. "This looks like an ordinary club website. The only thing
added is one line of code." Click Ask. Type: "How many overs can one bowler bowl?"
"It answers from their documents — and notice it handles the fact that the rules
differ by format, instead of giving one misleading number."

**[0:50–1:20] Show the trust features.**
Ask: "What are the most common cricket injuries?" Point at the cited sources.
"Every answer shows where it came from." Then ask something not in the docs:
"What was the sponsorship revenue?" "And when it doesn't know, it says so — it
doesn't make things up. That's the difference between a toy and something a
business can rely on."

**[1:20–1:50] Show it's theirs.**
"It runs on their documents, on their domain, behind a key they control, and it
embeds with a single line of code. I handle ingestion, the retrieval pipeline,
deployment, and the security setup."

**[1:50–2:00] Close.**
"If your team has documents people struggle to search, I build this as a
fixed-scope pilot. Details in the description."
