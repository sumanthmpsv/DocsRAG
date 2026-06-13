"""
Milestone 2 — Turn text into embeddings and search them.

An EMBEDDING is just text converted into a list of numbers (a vector) where
similar meaning lands at nearby points in space. To find passages relevant to a
question, we embed the question too, then find the chunk vectors closest to it.

"Closest" here = highest cosine similarity. We normalize every vector to length 1,
which makes cosine similarity equal to a plain dot product — so the whole search
is one matrix multiply (embeddings @ query). That's all a "vector database" does
under the hood; seeing it in ~5 lines of numpy is the point.

We keep the store in memory (a numpy array) for now so it runs with zero infra.
When we deploy, this is the one piece we swap for Postgres + pgvector — same idea,
just persisted and scalable.
"""
import pickle
import numpy as np
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"  # cheap, 1536-dim, plenty for this


class VectorStore:
    def __init__(self):
        self.client = OpenAI()
        self.embeddings: np.ndarray | None = None  # shape (N, d), L2-normalized
        self.metadata: list[dict] = []             # parallel list of chunk dicts

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for start in range(0, len(texts), 100):  # batch to stay within API limits
            batch = texts[start:start + 100]
            resp = self.client.embeddings.create(model=EMBED_MODEL, input=batch)
            vectors.extend(d.embedding for d in resp.data)
        arr = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-8, None)  # normalize -> cosine == dot product

    def build(self, chunks: list[dict]):
        self.metadata = chunks
        self.embeddings = self.embed_texts([c["text"] for c in chunks])

    def search(self, query: str, k: int = 4) -> list[dict]:
        q = self.embed_texts([query])[0]          # (d,)
        scores = self.embeddings @ q              # (N,) cosine similarities
        top = np.argsort(scores)[::-1][:k]        # indices of k highest scores
        results = []
        for idx in top:
            item = dict(self.metadata[idx])
            item["score"] = float(scores[idx])
            results.append(item)
        return results

    def save(self, path: str = "index.pkl"):
        with open(path, "wb") as f:
            pickle.dump({"embeddings": self.embeddings, "metadata": self.metadata}, f)

    @classmethod
    def load(cls, path: str = "index.pkl"):
        store = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        store.embeddings = data["embeddings"]
        store.metadata = data["metadata"]
        return store
