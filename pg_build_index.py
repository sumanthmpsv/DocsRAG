"""
Build the pgvector index from the docs/ folder.

    python pg_build_index.py            # uses ./docs
    python pg_build_index.py mydocs

Make sure the `chunks` table and `vector` extension exist first (see pg_store.py),
and that DATABASE_URL is set in .env. Reuses ingest.py unchanged — the cleaning
and chunking logic is identical; only the storage backend changed.
"""
import sys
from dotenv import load_dotenv
from ingest import load_documents
from pg_store import PgVectorStore

load_dotenv()


def main(folder: str = "docs"):
    chunks = load_documents(folder)
    if not chunks:
        print(f"No documents found in '{folder}/'. Add files and retry.")
        return
    print(f"Loaded {len(chunks)} chunks. Embedding and writing to Postgres...")
    store = PgVectorStore()
    store.build(chunks)
    print(f"Done. {store.count()} chunks now in the database.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs")
