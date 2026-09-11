"""
retriever.py
------------
Phase 4 (Vector Database) + Phase 5 (Retrieval).

Stores chunk embeddings in a FAISS index (IndexFlatIP, i.e. exact cosine
similarity search since embeddings are pre-normalized) and provides
Top-K semantic retrieval for a user query, including a similarity-score
threshold so low-relevance chunks can be filtered out before being sent
to the LLM (this directly supports hallucination handling in Phase 9).

Why FAISS for this project:
- Runs fully locally with no external service/API dependency
- IndexFlatIP gives exact (not approximate) nearest-neighbor search, which
  is fine at this document collection's scale (hundreds-thousands of
  chunks) and keeps retrieval quality easy to reason about
- Simple to persist to disk (index.faiss + a parallel metadata store)
"""

from __future__ import annotations

import os
import json
import pickle
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
try:
    import faiss
except Exception:
    faiss = None

from chunking import Chunk
from embeddings import EmbeddingModel


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_name: str
    page_number: int
    section: str
    text: str
    score: float  # cosine similarity, range [-1, 1], typically [0, 1]


class VectorStore:
    def __init__(self, embedding_model: Optional[EmbeddingModel] = None):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Chunk] = []

    # ------------------------------------------------------------------
    # Build / update
    # ------------------------------------------------------------------
    def build(self, chunks: List[Chunk]) -> None:
        """Embed all chunks and build a fresh FAISS index."""
        self.chunks = chunks
        texts = [c.text for c in chunks]
        vectors = self.embedding_model.embed_texts(texts)
        self.index = vectors

    def add(self, new_chunks: List[Chunk]) -> None:
        """
        Incrementally add new chunks to an existing index (supports the
        'document update / re-indexing' advanced feature without a full
        rebuild).
        """
        if self.index is None:
            self.build(new_chunks)
            return
        vectors = self.embedding_model.embed_texts([c.text for c in new_chunks])
        self.index.add(vectors)
        self.chunks.extend(new_chunks)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 4,
               score_threshold: float = 0.0) -> List[RetrievedChunk]:
        """
        Embed the query and retrieve the top_k most similar chunks.
        Chunks below `score_threshold` cosine similarity are dropped —
        this is the retrieval-confidence mechanism that feeds hallucination
        handling: if nothing clears the bar, the LLM is told there is no
        sufficient context rather than being handed weak/irrelevant chunks.
        """
        if self.index is None or len(self.index) == 0:
            return []

        query_vec = self.embedding_model.embed_query(query).reshape(-1)
        scores = np.dot(self.index, query_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[RetrievedChunk] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < score_threshold:
                continue
            chunk = self.chunks[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_name=chunk.doc_name,
                    page_number=chunk.page_number,
                    section=chunk.section,
                    text=chunk.text,
                    score=score,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, "index.npy"), self.index)
        with open(os.path.join(directory, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        meta = {
            "embedding_model": self.embedding_model.model_name,
            "num_chunks": len(self.chunks),
            "dimension": self.embedding_model.dimension,
        }
        with open(os.path.join(directory, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        with open(os.path.join(directory, "meta.json")) as f:
            meta = json.load(f)
        embedding_model = EmbeddingModel(meta["embedding_model"])
        store = cls(embedding_model=embedding_model)
        store.index = np.load(os.path.join(directory, "index.npy"))
        with open(os.path.join(directory, "chunks.pkl"), "rb") as f:
            store.chunks = pickle.load(f)
        return store


if __name__ == "__main__":
    from document_loader import load_documents
    from chunking import chunk_documents

    here = os.path.dirname(__file__)
    docs_dir = os.path.join(here, "..", "data", "documents")
    store_dir = os.path.join(here, "..", "vectorstore")

    records = load_documents(docs_dir)
    chunks = chunk_documents(records)

    store = VectorStore()
    store.build(chunks)
    store.save(store_dir)
    print(f"Built and saved vector store with {len(chunks)} chunks to {store_dir}")

    results = store.search("How many annual leave days do employees get?", top_k=3)
    for r in results:
        print(f"\n[{r.score:.3f}] {r.doc_name} p{r.page_number} {r.section}")
        print(r.text[:150], "...")
