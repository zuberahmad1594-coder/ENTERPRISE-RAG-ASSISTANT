"""
evaluation.py
-------------
Phase 18: RAG Evaluation.

Provides:
1. A small hand-labeled evaluation dataset (question, expected source
   document, reference answer) covering all four HR documents plus one
   deliberately out-of-scope question to test hallucination handling.
2. Retrieval evaluation: did the correct source document appear in the
   top-k retrieved chunks? (retrieval relevance / hit rate)
3. Basic answer evaluation helpers: keyword-based groundedness check and
   a simple correctness flag for manual review.

For a more advanced setup, this dataset can be fed into RAGAS
(https://github.com/explodinggecko/ragas) to compute faithfulness,
answer relevance, and context precision/recall automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import json
import os

from retriever import VectorStore
from rag_pipeline import answer_question, NO_ANSWER_MESSAGE


@dataclass
class EvalExample:
    question: str
    expected_source_doc: Optional[str]   # None means "should refuse to answer"
    reference_answer: str                # human-written reference, for manual grading
    notes: str = ""


EVAL_DATASET: List[EvalExample] = [
    EvalExample(
        question="How many annual leave days are available per year?",
        expected_source_doc="leave_policy.pdf",
        reference_answer="18 days of Annual Leave per calendar year, accrued at 1.5 days/month.",
    ),
    EvalExample(
        question="What is the resignation notice period for an Associate-level employee?",
        expected_source_doc="employee_handbook.pdf",
        reference_answer="60 days for Associate level and above.",
    ),
    EvalExample(
        question="How many days per week must employees come into the office under the hybrid model?",
        expected_source_doc="work_from_home_policy.pdf",
        reference_answer="A minimum of 3 days per week in the office by default.",
    ),
    EvalExample(
        question="What should an employee do if they witness workplace harassment?",
        expected_source_doc="code_of_conduct.pdf",
        reference_answer="Report it immediately to HR or the confidential ethics hotline.",
    ),
    EvalExample(
        question="How much is the internet reimbursement for remote work?",
        expected_source_doc="work_from_home_policy.pdf",
        reference_answer="Up to INR 1,500 per month with a valid bill.",
    ),
    EvalExample(
        question="What is the maximum value of a business gift an employee can accept without declaring it?",
        expected_source_doc="code_of_conduct.pdf",
        reference_answer="Gifts exceeding INR 2,000 must be declared to Compliance.",
    ),
    EvalExample(
        question="Does the company provide a company car to all employees?",
        expected_source_doc=None,  # not present anywhere in the corpus
        reference_answer=NO_ANSWER_MESSAGE,
        notes="Out-of-scope question; tests hallucination handling.",
    ),
    EvalExample(
        question="What is the probation period for new employees?",
        expected_source_doc="employee_handbook.pdf",
        reference_answer="90 days from the date of joining.",
    ),
]


@dataclass
class EvalResult:
    question: str
    expected_source_doc: Optional[str]
    retrieved_docs: List[str]
    retrieval_hit: bool          # expected doc found among retrieved chunks
    generated_answer: str
    correctly_refused: Optional[bool]  # only meaningful when expected_source_doc is None
    sources_cited: List[str]


def run_evaluation(store: VectorStore, top_k: int = 4) -> List[EvalResult]:
    results: List[EvalResult] = []

    for example in EVAL_DATASET:
        response = answer_question(store, example.question, top_k=top_k)
        retrieved_docs = list({c.doc_name for c in response.retrieved_chunks})

        if example.expected_source_doc is None:
            retrieval_hit = len(response.retrieved_chunks) == 0 or not response.grounded
            correctly_refused = not response.grounded
        else:
            retrieval_hit = example.expected_source_doc in retrieved_docs
            correctly_refused = None

        results.append(EvalResult(
            question=example.question,
            expected_source_doc=example.expected_source_doc,
            retrieved_docs=retrieved_docs,
            retrieval_hit=retrieval_hit,
            generated_answer=response.answer,
            correctly_refused=correctly_refused,
            sources_cited=response.sources,
        ))

    return results


def summarize(results: List[EvalResult]) -> dict:
    total = len(results)
    retrieval_hits = sum(1 for r in results if r.retrieval_hit)
    refusal_cases = [r for r in results if r.expected_source_doc is None]
    correct_refusals = sum(1 for r in refusal_cases if r.correctly_refused)

    return {
        "total_questions": total,
        "retrieval_hit_rate": round(retrieval_hits / total, 3) if total else 0,
        "hallucination_refusal_accuracy": (
            round(correct_refusals / len(refusal_cases), 3) if refusal_cases else None
        ),
    }


def save_results(results: List[EvalResult], path: str) -> None:
    data = [
        {
            "question": r.question,
            "expected_source_doc": r.expected_source_doc,
            "retrieved_docs": r.retrieved_docs,
            "retrieval_hit": r.retrieval_hit,
            "generated_answer": r.generated_answer,
            "correctly_refused": r.correctly_refused,
            "sources_cited": r.sources_cited,
        }
        for r in results
    ]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    store_dir = os.path.join(here, "..", "vectorstore")
    store = VectorStore.load(store_dir)

    results = run_evaluation(store)
    for r in results:
        status = "HIT" if r.retrieval_hit else "MISS"
        print(f"[{status}] {r.question}")
        print(f"   expected={r.expected_source_doc}  retrieved={r.retrieved_docs}")
        print(f"   answer={r.generated_answer[:100]}...\n")

    summary = summarize(results)
    print("=== Summary ===")
    print(json.dumps(summary, indent=2))

    out_path = os.path.join(here, "..", "data", "processed", "evaluation_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_results(results, out_path)
    print(f"\nSaved detailed results to {out_path}")
