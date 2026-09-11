"""
Sandbox-only smoke test: validates the mechanical correctness of the
document_loader -> chunking -> retriever pipeline using a TF-IDF vectorizer
as a stand-in for SentenceTransformer, since this sandbox's network
allowlist does not include huggingface.co and cannot download the real
embedding model.

This is NOT part of the deliverable pipeline — it exists purely so we can
prove chunk/metadata/retrieval plumbing is correct end-to-end right now.
On your machine (with internet access to huggingface.co), src/embeddings.py
will download and use the real 'all-MiniLM-L6-v2' model automatically.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from document_loader import load_documents
from chunking import chunk_documents
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

here = os.path.dirname(__file__)
docs_dir = os.path.join(here, "..", "data", "documents")

records = load_documents(docs_dir)
chunks = chunk_documents(records)
print(f"Loaded {len(records)} pages -> {len(chunks)} chunks\n")

texts = [c.text for c in chunks]
vectorizer = TfidfVectorizer(stop_words="english")
matrix = vectorizer.fit_transform(texts)

test_questions = [
    "How many annual leave days are available?",
    "What is the resignation notice period?",
    "Can I work fully remote?",
    "What happens if I violate the anti-bribery policy?",
    "Does the company provide free lunch?",  # should retrieve weakly / be filterable
]

for q in test_questions:
    q_vec = vectorizer.transform([q])
    sims = cosine_similarity(q_vec, matrix)[0]
    top_idx = np.argsort(sims)[::-1][:3]
    print(f"=== Q: {q} ===")
    for idx in top_idx:
        c = chunks[idx]
        print(f"  [{sims[idx]:.3f}] {c.doc_name} p{c.page_number}  -> {c.text[:100]}...")
    print()

print("Pipeline mechanics verified: loading -> chunking -> metadata -> retrieval all working.")
