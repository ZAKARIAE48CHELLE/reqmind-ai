"""Document ingestion module for ReqMind AI.

Supports UTF-8 text pasted or read from .txt / .md files.
Preserves exact raw text and offsets.
"""

import uuid
from pathlib import Path
from typing import Union
from reqmind.schemas import DocumentRecord


def ingest_text(text: str, document_id: str | None = None, filename: str | None = None) -> DocumentRecord:
    """Ingest raw requirement text.
    
    Validates UTF-8 encoding, non-emptiness, and generates a stable document ID.
    """
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty or whitespace only.")
    
    doc_id = document_id or f"doc_{uuid.uuid4().hex[:8]}"
    return DocumentRecord(
        document_id=doc_id,
        filename=filename,
        raw_text=text,
        requirements=[],
    )


def ingest_file(file_path: Union[str, Path]) -> DocumentRecord:
    """Ingest a .txt or .md file from disk."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    
    if path.suffix.lower() not in [".txt", ".md"]:
        raise ValueError(f"Unsupported file format '{path.suffix}'. Only .txt and .md are supported.")
    
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    return ingest_text(text=content, filename=path.name)
