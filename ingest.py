"""
Load documents, strip repeated page boilerplate, and split into chunks.

Includes the boilerplate-stripping fix: repeated headers/footers are removed
before chunking so they don't pollute embeddings.

NOTE: the BOILERPLATE_PATTERNS below are specific to the sample coaching manual.
For real client documents, replace this with a general repeated-line detector
(detect lines that recur at the same position across many pages and strip those).
That generalization is real billable work, not a one-liner.
"""
import re
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED = {".pdf", ".txt", ".md"}

BOILERPLATE_PATTERNS = [
    re.compile(r"RIVERSIDE CC.*?DO NOT DISTRIBUTE", re.IGNORECASE),
    re.compile(r"Revision \d{4}-\d{2}.*?Internal use only", re.IGNORECASE),
    re.compile(r"Page \d+", re.IGNORECASE),
]


def _strip_boilerplate(text: str) -> str:
    for pat in BOILERPLATE_PATTERNS:
        text = pat.sub("", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return _strip_boilerplate(text)
    return path.read_text(encoding="utf-8", errors="ignore")


def load_documents(folder: str = "docs") -> list[dict]:
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
