from pathlib import Path
from typing import Dict, List

import pdfplumber
from docx import Document as DocxDocument


def load_txt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_pdf(path: str | Path) -> str:
    parts: List[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                parts.append(text.strip())
    return "\n".join(parts).strip()


def load_docx(path: str | Path) -> str:
    doc = DocxDocument(str(path))
    parts = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(parts).strip()


def load_document(path: str | Path) -> Dict[str, str]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".txt":
        text = load_txt(p)
    elif suffix == ".pdf":
        text = load_pdf(p)
    elif suffix == ".docx":
        text = load_docx(p)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    return {"doc_id": p.stem, "title": p.name, "text": text}


def load_documents_from_dir(directory: str | Path) -> List[Dict[str, str]]:
    base = Path(directory)
    docs: List[Dict[str, str]] = []
    for pattern in ("*.txt", "*.pdf", "*.docx"):
        for path in sorted(base.glob(pattern)):
            loaded = load_document(path)
            if loaded["text"].strip():
                docs.append(loaded)
    return docs
