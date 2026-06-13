-- One-time database setup for DocsRAG.
-- Run after the database exists:  psql "$DATABASE_URL" -f schema.sql

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id        BIGSERIAL PRIMARY KEY,
    source    TEXT NOT NULL,
    position  INT  NOT NULL,
    text      TEXT NOT NULL,
    embedding VECTOR(1536) NOT NULL
);

-- HNSW index keeps cosine-similarity search fast as the table grows.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
