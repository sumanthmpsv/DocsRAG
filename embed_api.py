"""
Production embeddable API — pgvector-backed, with API-key auth and CORS lock.

Endpoints:
  GET  /                -> polished demo page (a mock client site with the widget)
  GET  /widget.js       -> the drop-in script a client pastes into their site
  POST /ask             -> answer endpoint (requires X-API-Key header)
  GET  /health          -> liveness check (no auth)

Security posture (the difference between a demo and a deployable product):
  - /ask requires an API key, set via DOCSRAG_API_KEY in the environment.
  - CORS is locked to ALLOWED_ORIGINS (comma-separated env var). Never ship "*".
  - A simple in-memory rate limit guards against runaway cost from a hammered key.
    (For multi-instance production, move the limiter to Redis.)

Env (.env):
  OPENAI_API_KEY=sk-...
  DATABASE_URL=postgresql://user:pass@host:5432/db
  DOCSRAG_API_KEY=choose-a-long-random-string
  ALLOWED_ORIGINS=https://acme.com,https://www.acme.com

Run:
  uvicorn embed_api:app --reload
"""
import os
import time
from collections import defaultdict, deque

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

import pg_rag
from pg_store import PgVectorStore

load_dotenv()

API_KEY = os.environ.get("DOCSRAG_API_KEY", "dev-key-change-me")
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")]
RATE_LIMIT = 30          # max requests
RATE_WINDOW = 60         # per this many seconds, per client IP

app = FastAPI(title="DocsRAG", description="Embeddable document Q&A with sources.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

_hits: dict[str, deque] = defaultdict(deque)


def _rate_ok(ip: str) -> bool:
    now = time.time()
    q = _hits[ip]
    while q and q[0] < now - RATE_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        return False
    q.append(now)
    return True


# One store/connection reused across requests.
try:
    STORE = PgVectorStore()
    _INDEXED = STORE.count() > 0
except Exception:
    STORE = None
    _INDEXED = False


class AskRequest(BaseModel):
    question: str
    k: int = 6


@app.get("/health")
def health():
    return {"status": "ok", "indexed": _INDEXED}


@app.post("/ask")
def ask(req: AskRequest, request: Request, x_api_key: str = Header(default="")):
    if x_api_key != API_KEY:
        raise HTTPException(401, "Invalid or missing API key.")
    if not _rate_ok(request.client.host):
        raise HTTPException(429, "Rate limit exceeded. Try again shortly.")
    if STORE is None:
        raise HTTPException(503, "Store unavailable. Check DATABASE_URL and run pg_build_index.py.")
    if not req.question.strip():
        raise HTTPException(400, "Question must not be empty.")
    return pg_rag.answer(req.question, k=req.k, store=STORE)


# The client embeds DocsRAG with one line. The key is baked into the snippet the
# client copies, so it lives in their page — fine for a public docs assistant,
# since the key only authorizes asking questions and is rate-limited.
WIDGET_JS = """
(function () {
  var me = document.currentScript || document.querySelector('script[data-docsrag]');
  var base = me ? me.src.replace(/\\/widget\\.js.*$/, '') : '';
  var key = me ? (me.getAttribute('data-key') || '') : '';

  var btn = document.createElement('button');
  btn.textContent = 'Ask';
  btn.setAttribute('aria-label', 'Ask about our documents');
  btn.style.cssText = 'position:fixed;bottom:20px;right:20px;z-index:99999;border:none;'
    + 'border-radius:999px;padding:12px 22px;font:600 14px system-ui,sans-serif;'
    + 'background:#16504B;color:#fff;cursor:pointer;box-shadow:0 6px 20px rgba(0,0,0,.22)';

  var panel = document.createElement('div');
  panel.setAttribute('role', 'dialog');
  panel.style.cssText = 'position:fixed;bottom:74px;right:20px;z-index:99999;width:340px;'
    + 'max-width:92vw;background:#fff;border:1px solid #e5e7eb;border-radius:14px;'
    + 'box-shadow:0 12px 40px rgba(0,0,0,.18);padding:16px;font:14px system-ui,sans-serif;display:none';
  panel.innerHTML =
      '<div style="font-weight:600;margin-bottom:8px;color:#16504B">Ask our docs</div>'
    + '<input id="dr-q" placeholder="Type a question..." aria-label="Your question" '
    + 'style="width:100%;box-sizing:border-box;padding:9px;border:1px solid #cbd5e1;border-radius:8px"/>'
    + '<button id="dr-send" style="margin-top:8px;width:100%;padding:9px;border:none;border-radius:8px;'
    + 'background:#16504B;color:#fff;cursor:pointer;font-weight:600">Send</button>'
    + '<div id="dr-out" style="margin-top:12px;white-space:pre-wrap;line-height:1.5"></div>';

  document.body.appendChild(btn);
  document.body.appendChild(panel);
  btn.onclick = function () {
    panel.style.display = (panel.style.display === 'none') ? 'block' : 'none';
    if (panel.style.display === 'block') document.getElementById('dr-q').focus();
  };

  function ask() {
    var q = document.getElementById('dr-q').value.trim();
    var out = document.getElementById('dr-out');
    if (!q) return;
    out.textContent = 'Thinking...';
    fetch(base + '/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-API-Key': key },
      body: JSON.stringify({ question: q })
    })
      .then(function (r) {
        if (r.status === 429) throw new Error('Too many questions just now. Please wait a moment.');
        if (!r.ok) throw new Error('Something went wrong. Please try again.');
        return r.json();
      })
      .then(function (d) {
        var srcs = (d.sources || []).map(function (s) {
          return '\\u2022 ' + s.source + ' #' + s.position;
        }).join('\\n');
        out.textContent = d.answer + (srcs ? '\\n\\nSources:\\n' + srcs : '');
      })
      .catch(function (e) { out.textContent = e.message; });
  }
  panel.querySelector('#dr-send').onclick = ask;
  panel.querySelector('#dr-q').addEventListener('keydown', function (e) {
    if (e.key === 'Enter') ask();
  });
})();
"""


@app.get("/widget.js")
def widget_js():
    return Response(content=WIDGET_JS, media_type="application/javascript")


@app.get("/", response_class=HTMLResponse)
def demo_page():
    key = API_KEY
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Riverside CC — Members' Hub</title>
<style>
  :root {{ --ink:#14201e; --field:#16504B; --line:#e7e4dd; --paper:#faf8f3; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--paper); color:var(--ink);
    font:17px/1.65 Georgia, 'Times New Roman', serif; }}
  .wrap {{ max-width:680px; margin:0 auto; padding:72px 24px 120px; }}
  .eyebrow {{ font:600 12px/1 system-ui,sans-serif; letter-spacing:.18em;
    text-transform:uppercase; color:var(--field); margin-bottom:18px; }}
  h1 {{ font-size:44px; line-height:1.05; margin:0 0 8px; letter-spacing:-.01em; }}
  .lede {{ font-size:20px; color:#4a5550; margin:0 0 40px; }}
  h2 {{ font:600 13px/1 system-ui,sans-serif; letter-spacing:.14em;
    text-transform:uppercase; color:var(--field); margin:40px 0 10px;
    padding-bottom:8px; border-bottom:1px solid var(--line); }}
  p {{ margin:0 0 16px; }}
  .hint {{ font:15px/1.6 system-ui,sans-serif; background:#fff; border:1px solid var(--line);
    border-left:3px solid var(--field); border-radius:8px; padding:14px 16px; color:#3a443f; }}
  code {{ font:14px ui-monospace, Menlo, monospace; background:#fff;
    border:1px solid var(--line); border-radius:5px; padding:1px 6px; }}
</style></head>
<body>
  <div class="wrap">
    <div class="eyebrow">Riverside Cricket Club</div>
    <h1>Members' Hub</h1>
    <p class="lede">Fixtures, the laws, club rules, and the coaching manual —
       all in one place.</p>

    <h2>This is a demo</h2>
    <p>This page is a stand-in for a client's own website. The only thing added
       is a single line at the bottom of the page source — that one line embeds
       the assistant. The <strong>Ask</strong> button in the lower-right answers
       questions from the club's indexed documents, and cites where each answer
       came from.</p>

    <div class="hint">Try: <em>"How many overs can one bowler bowl?"</em> or
       <em>"What are the most common cricket injuries?"</em></div>

    <h2>Why it lives here</h2>
    <p>The assistant runs on the club's own documents, on the club's own page,
       behind a key the club controls. It is not a general chatbot pointed at the
       public internet — it answers only from what it has been given.</p>
  </div>

  <script src="/widget.js" data-docsrag data-key="{key}"></script>
</body></html>"""
