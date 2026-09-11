"""
chunking.py
-----------
Phase 2: Text Chunking.

Splits each page's cleaned text into overlapping, fixed-size chunks
(measured in words), attempting to break on sentence boundaries where
possible so chunks remain semantically coherent. Each chunk retains full
metadata (document name, page number, chunk id, section heading if
detectable) so it can be cited later.

Design notes (documented for the README / demo Q&A):
- Chunk size: 220 words, overlap: 40 words (~18% overlap) by default.
  This range keeps chunks small enough for precise retrieval but large
  enough to preserve context for a full policy clause.
- Overlap prevents a fact from being silently split across the boundary
  of two chunks, which would otherwise reduce retrieval recall.
- Extremely small chunks (e.g., single sentences) lose surrounding
  context and increase the number of vectors to search without adding
  useful signal. Extremely large chunks (whole pages) reduce retrieval
  precision because the embedding represents many unrelated ideas at
  once, diluting the similarity signal for a specific question.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from document_loader import PageRecord


@dataclass
class Chunk:
    chunk_id: str
    doc_name: str
    page_number: int
    section: str
    text: str


_SECTION_HEADING_RE = re.compile(r"^\s*\d{1,2}\.\s+[A-Z][A-Za-z /\-&]+")


def _detect_section(text: str) -> str:
    """Try to find a section heading like '3. Sick Leave' at the start of text."""
    first_line = text.split("\n", 1)[0]
    if _SECTION_HEADING_RE.match(first_line):
        return first_line.strip()
    return ""


def _split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter (avoids adding an nltk dependency)."""
    # Split on '.', '?', '!' followed by whitespace and a capital letter/newline
    sentences = re.split(r"(?<=[.?!])\s+(?=[A-Z0-9\n])", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_page(record: PageRecord, chunk_size_words: int = 220,
               overlap_words: int = 40) -> List[Chunk]:
    """
    Chunk a single PageRecord into one or more overlapping Chunks,
    breaking on sentence boundaries so we don't cut a sentence in half.
    """
    sentences = _split_sentences(record.text)
    section = _detect_section(record.text)

    chunks: List[Chunk] = []
    current_words: List[str] = []
    chunk_index = 0

    def flush(words: List[str]):
        nonlocal chunk_index
        if not words:
            return
        chunk_text = " ".join(words)
        chunk_id = f"{record.doc_name}::p{record.page_number}::c{chunk_index}"
        chunks.append(Chunk(
            chunk_id=chunk_id,
            doc_name=record.doc_name,
            page_number=record.page_number,
            section=section,
            text=chunk_text,
        ))
        chunk_index += 1

    for sentence in sentences:
        sentence_words = sentence.split()
        if len(current_words) + len(sentence_words) > chunk_size_words and current_words:
            flush(current_words)
            # Start next chunk with overlap from the tail of the previous chunk
            overlap = current_words[-overlap_words:] if overlap_words else []
            current_words = overlap + sentence_words
        else:
            current_words.extend(sentence_words)

    flush(current_words)
    return chunks


def chunk_documents(records: List[PageRecord], chunk_size_words: int = 220,
                     overlap_words: int = 40) -> List[Chunk]:
    """Chunk every page record in the collection."""
    all_chunks: List[Chunk] = []
    for record in records:
        all_chunks.extend(chunk_page(record, chunk_size_words, overlap_words))
    return all_chunks


if __name__ == "__main__":
    from document_loader import load_documents
    import os

    here = os.path.dirname(__file__)
    docs_dir = os.path.join(here, "..", "data", "documents")
    records = load_documents(docs_dir)
    chunks = chunk_documents(records)

    print(f"Loaded {len(records)} pages -> produced {len(chunks)} chunks")
    for c in chunks[:3]:
        print(f"\n[{c.chunk_id}] section='{c.section}'")
        print(c.text[:180], "...")
