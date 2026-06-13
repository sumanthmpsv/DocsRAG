"""
RAG answer logic — production version using the pgvector store.

Identical retrieval-and-ground logic to the local rag.py; only the store type
changed. Keeps the conflict-surfacing prompt and k=6 default you tuned.
"""
from dotenv import load_dotenv
from openai import OpenAI
from pg_store import PgVectorStore

load_dotenv()

CHAT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided context passages. "
    "If the answer is not contained in the context, say "
    "\"I don't know based on the provided documents.\" "
    "If the context contains conflicting or differing information, do NOT pick "
    "one and hide the rest. Instead, surface each version, attribute it to its "
    "source, and state the condition under which each applies (for example a "
    "specific format, league, or section). "
    "Be concise and never invent facts."
)


def _format_context(results: list[dict]) -> str:
    return "\n\n".join(
        f"[Source {i}: {r['source']} #{r['position']}]\n{r['text']}"
        for i, r in enumerate(results, 1)
    )


def answer(question: str, k: int = 6, store: PgVectorStore | None = None) -> dict:
    store = store or PgVectorStore()
    results = store.search(question, k=k)
    context = _format_context(results)

    client = OpenAI()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return {
        "answer": resp.choices[0].message.content,
        "sources": [
            {
                "source": r["source"],
                "position": r["position"],
                "score": round(r["score"], 3),
                "snippet": r["text"][:200],
            }
            for r in results
        ],
    }
