"""
Sandbox-only test: validates retrieval-hit-rate logic from evaluation.py
against the EVAL_DATASET questions, using TF-IDF similarity as a stand-in
for the real sentence-transformer embeddings (since huggingface.co is not
reachable in this sandbox). This proves the retrieval side of the
evaluation harness is correct; the real run (with real embeddings + a live
ANTHROPIC_API_KEY) is what you'll execute in your own environment via
`python src/evaluation.py`.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from document_loader import load_documents
from chunking import chunk_documents
from evaluation import EVAL_DATASET
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

here = os.path.dirname(__file__)
docs_dir = os.path.join(here, "..", "data", "documents")
records = load_documents(docs_dir)
chunks = chunk_documents(records)
texts = [c.text for c in chunks]

vectorizer = TfidfVectorizer(stop_words="english")
matrix = vectorizer.fit_transform(texts)

TOP_K = 4
SCORE_THRESHOLD = 0.08  # TF-IDF scores run lower than semantic embeddings

hits = 0
refusal_correct = 0
refusal_total = 0

for ex in EVAL_DATASET:
    q_vec = vectorizer.transform([ex.question])
    sims = cosine_similarity(q_vec, matrix)[0]
    top_idx = np.argsort(sims)[::-1][:TOP_K]
    retrieved = [(chunks[i].doc_name, sims[i]) for i in top_idx if sims[i] >= SCORE_THRESHOLD]
    retrieved_docs = list({d for d, s in retrieved})

    if ex.expected_source_doc is None:
        refusal_total += 1
        correctly_refused = len(retrieved) == 0
        if correctly_refused:
            refusal_correct += 1
        status = "REFUSE-OK" if correctly_refused else "REFUSE-FAIL(would hallucinate)"
    else:
        hit = ex.expected_source_doc in retrieved_docs
        hits += 1 if hit else 0
        status = "HIT" if hit else "MISS"

    print(f"[{status}] {ex.question}")
    print(f"    expected={ex.expected_source_doc}  retrieved_docs={retrieved_docs}  top_scores={[round(s,3) for _,s in retrieved][:3]}")

non_refusal_total = len(EVAL_DATASET) - refusal_total
print(f"\nRetrieval hit rate (answerable qs): {hits}/{non_refusal_total} = {hits/non_refusal_total:.2f}")
if refusal_total:
    print(f"Correct refusal rate (out-of-scope qs): {refusal_correct}/{refusal_total} = {refusal_correct/refusal_total:.2f}")
