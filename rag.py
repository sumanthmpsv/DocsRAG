"""
Milestone 3 — Retrieval-Augmented Generation, the core idea.

The whole trick: instead of asking the model to answer from memory (where it can
hallucinate), we (1) retrieve the most relevant chunks from our own documents,
(2) paste them into the prompt as context, and (3) instruct the model to answer
ONLY from that context. The sources we return are exactly the chunks we fed in,
so every answer is traceable.

temperature=0 makes answers as deterministic and grounded as possible.

Want Claude instead of OpenAI for the answer step? Swap the client + call for the
Anthropic SDK; the retrieval half stays identical. Keeping one provider for now
so you only need one API key.
"""
from dotenv import load_dotenv
from openai import OpenAI
from store import VectorStore

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


def answer(question: str, k: int = 6, store: VectorStore | None = None) -> dict:
    store = store or VectorStore.load("index.pkl")
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


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What is this document about?"
    out = answer(q)
    print("\nQ:", q)
    print("\nA:", out["answer"])
    print("\nSources:")
    for s in out["sources"]:
        print(f"  - {s['source']} #{s['position']} (score {s['score']}): {s['snippet'][:80]}...")
