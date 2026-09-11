"""
embeddings.py
-------------
Phase 3: Embeddings.

Wraps a Sentence-Transformers model to convert document chunks and user
queries into dense vector representations. The SAME model is used for both
documents and queries, which is required: embeddings from different models
live in different vector spaces and are not comparable via cosine similarity.

Model: 'all-MiniLM-L6-v2'
- 384-dimensional embeddings
- Fast, CPU-friendly, strong general-purpose semantic similarity performance
- Good default for a capstone project; can be swapped for a larger model
  (e.g., 'all-mpnet-base-v2') for higher quality at the cost of speed.
"""

from __future__ import annotations

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingModel:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of document chunk texts.
        Returns an (N, dim) float32 numpy array, L2-normalized so that
        cosine similarity == dot product (required for FAISS IndexFlatIP).
        """
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single user query. Returns a (dim,) float32 vector, normalized."""
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.astype("float32")[0]


if __name__ == "__main__":
    model = EmbeddingModel()
    print(f"Loaded {model.model_name} | dimension={model.dimension}")

    sample_texts = [
        "Employees are entitled to 18 days of annual leave per year.",
        "The company provides health insurance to full-time staff.",
    ]
    vecs = model.embed_texts(sample_texts)
    print("Chunk embeddings shape:", vecs.shape)

    q_vec = model.embed_query("How many annual leave days do I get?")
    print("Query embedding shape:", q_vec.shape)

    # Cosine similarity via dot product (vectors are normalized)
    sims = vecs @ q_vec
    print("Similarities:", sims)
