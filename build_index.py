"""
Build step — chunk every document, embed the chunks, and save the index to disk.

Run this once after adding/changing files in docs/:

    python build_index.py            # uses ./docs
    python build_index.py mydocs     # uses ./mydocs

This is the first moment you "see it working": it prints how many chunks were
embedded and the vector dimension. The API and the Streamlit UI both just load
the index.pkl this produces, so they start instantly without re-embedding.
"""
import sys
from dotenv import load_dotenv
from ingest import load_documents
from store import VectorStore

load_dotenv()


def main(folder: str = "docs"):
    chunks = load_documents(folder)
    if not chunks:
        print(f"No documents found in '{folder}/'. Add .pdf, .txt, or .md files and retry.")
        return
    print(f"Loaded {len(chunks)} chunks. Embedding now (this calls the OpenAI API)...")
    store = VectorStore()
    store.build(chunks)
    store.save("index.pkl")
    print(
        f"Done. Embedded {len(chunks)} chunks -> index.pkl "
        f"(each vector has {store.embeddings.shape[1]} dimensions)."
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs")
