"""
Milestone 1 — Load documents and split them into chunks.

Why chunk at all? LLMs have a limited context window, and retrieval works far
better on small focused passages than on whole documents. So we slice each file
into overlapping ~800-character chunks. The overlap (120 chars) keeps sentences
that straddle a boundary from being cut in half.

This file calls NO external API — it's pure text processing, so run it first and
inspect the output before spending anything on embeddings.
"""
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED = {".pdf", ".txt", ".md"}


def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def load_documents(folder: str = "docs") -> list[dict]:
    """Return a list of {source, position, text} chunks from every file in `folder`."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks: list[dict] = []
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() not in SUPPORTED:
            continue
        text = _read_file(path)
        for i, chunk in enumerate(splitter.split_text(text)):
            if chunk.strip():
                chunks.append({"source": path.name, "position": i, "text": chunk.strip()})
    return chunks


if __name__ == "__main__":
    docs = load_documents()
    files = sorted({d["source"] for d in docs})
    print(f"Loaded {len(docs)} chunks from {len(files)} file(s): {files}")
    if docs:
        print("\n--- First chunk ---")
        print(docs[0]["text"][:300])
