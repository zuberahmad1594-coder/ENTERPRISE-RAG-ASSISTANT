"""
rag_pipeline.py
----------------
Phase 6 (Prompt Engineering) + Phase 7 (LLM Integration) +
Phase 8 (Source Attribution) + Phase 9 (Hallucination Handling) +
Phase 10 (Conversational Memory).

Ties retrieval to generation:
    question -> retrieve top-k chunks -> build grounded prompt
             -> call LLM -> parse answer + attach sources
             -> if no chunks / low confidence -> refuse gracefully

LLM provider: Anthropic Claude via the official `anthropic` Python SDK.
Set the ANTHROPIC_API_KEY environment variable before running.
You may swap in any other approved provider (OpenAI, Gemini, a local HF
model, etc.) by replacing `_call_llm` — the rest of the pipeline is
provider-agnostic.
"""

from __future__ import annotations

import os
from google import genai
from dataclasses import dataclass, field
from typing import List, Optional

from retriever import VectorStore, RetrievedChunk

NO_ANSWER_MESSAGE = (
    "I could not find sufficient information in the provided documents to "
    "answer this question."
)

# Minimum cosine similarity a chunk must clear to be considered "relevant
# enough" to ground an answer. Anything below this is treated as noise.
DEFAULT_SCORE_THRESHOLD = 0.30

SYSTEM_PROMPT = """You are an enterprise HR knowledge assistant for NorthBridge Solutions.

Your job is to answer employee questions using ONLY the context provided below,
which was retrieved from the company's official HR policy documents.

Rules you MUST follow:
1. Base your answer strictly on the provided context. Do not use outside
   knowledge, assumptions, or information not present in the context.
2. If the context does not contain enough information to answer the
   question, respond exactly with: "{no_answer}"
3. Do not fabricate policy numbers, dates, durations, or figures. Only
   state figures that literally appear in the context.
4. Keep answers concise, professional, and directly responsive to the
   question. Use bullet points only when listing multiple distinct items.
5. Do not mention "the context" or "the documents" explicitly in your
   answer — write as if you are simply answering from company policy
   knowledge, but never go beyond what the context supports.
6. If the user asks a follow-up question, use the conversation history to
   understand what they are referring to, but still ground the factual
   content of your answer only in the newly retrieved context provided
   for this turn.
""".format(no_answer=NO_ANSWER_MESSAGE)


@dataclass
class ConversationTurn:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class RAGResponse:
    answer: str
    sources: List[str]
    retrieved_chunks: List[RetrievedChunk]
    grounded: bool  # False if we returned the "no sufficient info" fallback


def _format_context(chunks: List[RetrievedChunk]) -> str:
    """Build the context block injected into the prompt, each chunk tagged
    with a citation-ready label so the model (and our own attribution
    logic) can reference exactly where it came from."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        section_part = f" — {c.section}" if c.section else ""
        label = f"[Source {i}: {c.doc_name}, Page {c.page_number}{section_part}]"
        blocks.append(f"{label}\n{c.text}")
    return "\n\n".join(blocks)


def _format_sources(chunks: List[RetrievedChunk]) -> List[str]:
    """De-duplicated, human-readable source references for display in the UI."""
    seen = set()
    sources = []
    for c in chunks:
        section_part = f" — {c.section}" if c.section else ""
        label = f"{c.doc_name} — Page {c.page_number}{section_part}"
        if label not in seen:
            seen.add(label)
            sources.append(label)
    return sources


def _call_llm(system_prompt: str, user_prompt: str, history: list = None) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    full_prompt = f"{system_prompt}\n\nUser Question: {user_prompt}"
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=full_prompt,
    )
    return response.text


def answer_question(
    store: VectorStore,
    question: str,
    top_k: int = 4,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    history: Optional[List[ConversationTurn]] = None,
) -> RAGResponse:
    """
    Full RAG turn: retrieve -> build grounded prompt -> generate -> attribute.
    """
    retrieved = store.search(question, top_k=top_k, score_threshold=score_threshold)

    # Hallucination handling: no relevant chunks -> refuse immediately,
    # without ever calling the LLM on empty/irrelevant context.
    if not retrieved:
        return RAGResponse(
            answer=NO_ANSWER_MESSAGE,
            sources=[],
            retrieved_chunks=[],
            grounded=False,
        )

    context_block = _format_context(retrieved)
    user_prompt = (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        f"Answer the question using only the context above."
    )

    answer_text = _call_llm(SYSTEM_PROMPT, user_prompt, history=history)
    grounded = NO_ANSWER_MESSAGE not in answer_text

    return RAGResponse(
        answer=answer_text,
        sources=_format_sources(retrieved) if grounded else [],
        retrieved_chunks=retrieved,
        grounded=grounded,
    )


if __name__ == "__main__":
    import os as _os
    here = _os.path.dirname(__file__)
    store_dir = _os.path.join(here, "..", "vectorstore")

    store = VectorStore.load(store_dir)

    test_questions = [
        "How many annual leave days are available?",
        "What is the resignation notice period?",
        "Does the company offer stock options?",  # expected: no-answer fallback
    ]

    for q in test_questions:
        print(f"\n=== Q: {q} ===")
        result = answer_question(store, q)
        print("A:", result.answer)
        print("Sources:", result.sources)
