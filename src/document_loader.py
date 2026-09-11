"""
document_loader.py
-------------------
Phase 1: Document Processing.

Loads PDF and DOCX files from a directory, extracts raw text page-by-page
(or paragraph-by-paragraph for DOCX), cleans whitespace/formatting noise,
and attaches document-level and page-level metadata that is preserved all
the way through chunking, retrieval, and source attribution.
"""

from __future__ import annotations

import os
import re
import glob
from dataclasses import dataclass, field
from typing import List

from pypdf import PdfReader
import docx


@dataclass
class PageRecord:
    """One page (PDF) or one 'section' (DOCX) of raw, cleaned text."""
    doc_name: str
    doc_path: str
    page_number: int          # 1-indexed; for DOCX we use a synthetic section number
    text: str


def _clean_text(text: str) -> str:
    """
    Basic text cleanup:
    - Normalize line breaks and collapse excessive whitespace
    - Strip common header/footer noise (page numbers, repeated titles)
    - Remove non-printable characters
    """
    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Drop lines that are just page numbers or very short boilerplate
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Skip lone page-number lines like "3" or "Page 3"
        if re.fullmatch(r"(page\s*)?\d{1,4}", stripped, flags=re.IGNORECASE):
            continue
        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)

    # Collapse multiple spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ newlines into 2 (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove non-printable / control characters
    text = re.sub(r"[^\x09\x0A\x20-\x7E]", "", text)

    return text.strip()


def load_pdf(path: str) -> List[PageRecord]:
    """Extract cleaned text from each page of a PDF."""
    doc_name = os.path.basename(path)
    reader = PdfReader(path)
    records: List[PageRecord] = []

    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        cleaned = _clean_text(raw)
        if cleaned:
            records.append(PageRecord(doc_name=doc_name, doc_path=path,
                                       page_number=i, text=cleaned))
    return records


def load_docx(path: str, paras_per_section: int = 12) -> List[PageRecord]:
    """
    DOCX files have no native 'page' concept, so we group paragraphs into
    synthetic sections of `paras_per_section` paragraphs each, and treat
    each section like a 'page' for downstream metadata consistency.
    """
    doc_name = os.path.basename(path)
    document = docx.Document(path)
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]

    records: List[PageRecord] = []
    section_num = 1
    for i in range(0, len(paragraphs), paras_per_section):
        chunk_paras = paragraphs[i:i + paras_per_section]
        raw = "\n".join(chunk_paras)
        cleaned = _clean_text(raw)
        if cleaned:
            records.append(PageRecord(doc_name=doc_name, doc_path=path,
                                       page_number=section_num, text=cleaned))
            section_num += 1
    return records


def load_documents(directory: str) -> List[PageRecord]:
    """
    Load every supported file (.pdf, .docx) in `directory`.
    Returns a flat list of PageRecord objects across all documents.
    """
    all_records: List[PageRecord] = []

    pdf_paths = sorted(glob.glob(os.path.join(directory, "*.pdf")))
    docx_paths = sorted(glob.glob(os.path.join(directory, "*.docx")))

    for path in pdf_paths:
        try:
            all_records.extend(load_pdf(path))
        except Exception as e:
            print(f"[document_loader] Failed to load PDF {path}: {e}")

    for path in docx_paths:
        try:
            all_records.extend(load_docx(path))
        except Exception as e:
            print(f"[document_loader] Failed to load DOCX {path}: {e}")

    return all_records


if __name__ == "__main__":
    # Quick manual test
    here = os.path.dirname(__file__)
    docs_dir = os.path.join(here, "..", "data", "documents")
    records = load_documents(docs_dir)
    print(f"Loaded {len(records)} page records from {docs_dir}")
    for r in records[:3]:
        print(f"\n--- {r.doc_name} | page {r.page_number} ---")
        print(r.text[:200], "...")
