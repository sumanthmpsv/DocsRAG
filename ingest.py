"""
Milestone 1 — Load documents and split them into chunks.

Why chunk at all? LLMs have a limited context window, and retrieval works far
better on small focused passages than on whole documents. So we slice each file
into overlapping ~800-character chunks. The overlap (120 chars) keeps sentences
that straddle a boundary from being cut in half.

This file calls NO external API — it's pure text processing, so run it first and
inspect the output before spending anything on embeddings.
"""
import re
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED = {".pdf", ".txt", ".md"}

# Lines that repeat on every page (headers/footers) carry no meaning and
# poison both the embeddings and the chunk boundaries. Strip them before chunking.
BOILERPLATE_PATTERNS = [
    re.compile(r"RIVERSIDE CC.*?DO NOT DISTRIBUTE", re.IGNORECASE),
    re.compile(r"Revision \d{4}-\d{2}.*?Internal use only", re.IGNORECASE),
    re.compile(r"Page \d+", re.IGNORECASE),
]


def _strip_boilerplate(text: str) -> str:
    for pat in BOILERPLATE_PATTERNS:
        text = pat.sub("", text)
    # collapse the blank lines left behind so spacing doesn't skew the splitter
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return _strip_boilerplate(text)
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
