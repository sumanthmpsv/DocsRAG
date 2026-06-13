"""
Production store — Postgres + pgvector.

Same idea as the in-memory store, now persisted and scalable. The search that was
`embeddings @ query` in NumPy becomes a SQL `ORDER BY embedding <=> query` — the
`<=>` operator is pgvector's cosine distance. The database does the math; you do
the plumbing. This is the swap that makes DocsRAG deployable and is squarely your
backend wheelhouse.

Requires:
    pip install pgvector psycopg[binary]
    DATABASE_URL in .env  (e.g. postgresql://user:pass@host:5432/dbname)

Set up the table once (psql or any client):
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE IF NOT EXISTS chunks (
        id        BIGSERIAL PRIMARY KEY,
        source    TEXT NOT NULL,
        position  INT  NOT NULL,
        text      TEXT NOT NULL,
        embedding VECTOR(1536) NOT NULL
    );
    CREATE INDEX IF NOT EXISTS chunks_embedding_idx
        ON chunks USING hnsw (embedding vector_cosine_ops);

The HNSW index keeps search fast as the table grows.
"""
import os
import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536


class PgVectorStore:
    def __init__(self, dsn: str | None = None):
        self.client = OpenAI()
        self.dsn = dsn or os.environ["DATABASE_URL"]
        self.conn = psycopg.connect(self.dsn, autocommit=True)
        register_vector(self.conn)

    def _embed(self, texts: list[str]) -> list[np.ndarray]:
        vectors = []
        for start in range(0, len(texts), 100):
            batch = texts[start:start + 100]
            resp = self.client.embeddings.create(model=EMBED_MODEL, input=batch)
            vectors.extend(np.array(d.embedding, dtype=np.float32) for d in resp.data)
        return vectors

    def build(self, chunks: list[dict]):
        """Embed and insert chunks. Clears existing rows first for a clean rebuild."""
        vectors = self._embed([c["text"] for c in chunks])
        with self.conn.cursor() as cur:
            cur.execute("TRUNCATE chunks RESTART IDENTITY;")
            for c, v in zip(chunks, vectors):
                cur.execute(
                    "INSERT INTO chunks (source, position, text, embedding) "
                    "VALUES (%s, %s, %s, %s)",
                    (c["source"], c["position"], c["text"], v),
                )

    def search(self, query: str, k: int = 6) -> list[dict]:
        qv = self._embed([query])[0]
        with self.conn.cursor() as cur:
            # 1 - cosine_distance = cosine_similarity, so we report a comparable score
            cur.execute(
                "SELECT source, position, text, 1 - (embedding <=> %s) AS score "
                "FROM chunks ORDER BY embedding <=> %s LIMIT %s",
                (qv, qv, k),
            )
            rows = cur.fetchall()
        return [
            {"source": r[0], "position": r[1], "text": r[2], "score": float(r[3])}
            for r in rows
        ]

    def count(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks;")
            return cur.fetchone()[0]
