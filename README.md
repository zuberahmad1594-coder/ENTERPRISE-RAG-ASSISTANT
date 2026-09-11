# Enterprise Document Intelligence \& RAG Assistant

An enterprise knowledge assistant that answers natural-language questions about a company's HR policy documents using Retrieval-Augmented Generation (RAG) — with cited sources and honest refusal when the documents don't cover the question.

**AI Capstone Project** — End-to-End Generative AI Application

\---

## 1\. Business Problem

Organizations store critical policy information across many separate PDFs and manuals — employee handbooks, leave policies, WFH policies, codes of conduct. Employees waste time searching multiple documents to answer simple questions ("How many leave days do I get?", "Can I work fully remote?"), and HR teams field the same repetitive questions daily.

**Core question:** Can an AI system understand an organization's document collection and provide accurate, source-grounded answers — without inventing information that isn't there?

## 2\. Selected Domain: HR Intelligence Assistant

This project was built around **HR policy documents** because:

* HR policy questions are high-frequency, repetitive, and well-suited to grounded Q\&A
* The domain naturally spans multiple related documents (handbook, leave policy, WFH policy, code of conduct), which is exactly the "coherent multi-document collection" the project calls for
* Wrong or hallucinated answers to HR questions (notice periods, leave entitlements) have real consequences, making hallucination handling and source attribution especially important to demonstrate

### Document Sources \& Usage Conditions

The four documents in `data/documents/` (`employee\_handbook.pdf`, `leave\_policy.pdf`, `work\_from\_home\_policy.pdf`, `code\_of\_conduct.pdf`) are **original, self-authored documents** written for a fictional company ("NorthBridge Solutions") specifically for this project. They are not copied from any real organization, so there are no confidentiality or copyright concerns. They were generated via `scripts/generate\_sample\_docs.py` and can be freely replaced with your own organization's public or permissioned documents — the pipeline works with any PDF/DOCX collection.

## 3\. System Architecture

```
Documents (PDF/DOCX)
        |
        v
Document Loading \& Text Extraction  (src/document\_loader.py)
        |
        v
Text Cleaning (whitespace, page-number noise, control chars)
        |
        v
Chunking + Metadata  (src/chunking.py)
        |
        v
Embeddings  (src/embeddings.py — Sentence-Transformers, all-MiniLM-L6-v2)
        |
        v
Vector Database  (src/retriever.py — FAISS IndexFlatIP)
        |
        |  User Question
        |       |
        |       v
        +---> Question Embedding (same model as documents)
                |
                v
          Semantic Retrieval (Top-K + similarity threshold)
                |
                v
          Relevant Context
                |
                v
          LLM (Claude, via Anthropic API)  (src/rag\_pipeline.py)
                |
                v
          Grounded Answer + Source References
                |
                v
          Streamlit UI  (app.py)
```

## 4\. Document Processing

* **Extraction:** `pypdf` for PDFs (page-by-page), `python-docx` for Word documents (grouped into synthetic "sections" of \~12 paragraphs each, since DOCX has no native page concept)
* **Cleaning:** normalizes line endings, strips lone page-number lines, collapses repeated whitespace/newlines, removes non-printable characters
* **Metadata preserved:** document name, page/section number, detected section heading (e.g., "3. Sick Leave"), and a unique chunk ID (`doc::page::chunk\_index`) — carried through to the final answer's source citations

## 5\. Chunking Strategy

* **Chunk size:** 220 words, **overlap:** 40 words, splitting on sentence boundaries (never mid-sentence)
* **Why this size:** tested 100/220/400-word configurations in `notebooks/02\_chunking\_and\_embeddings.ipynb`. Chunks of 100 words fragmented single policy clauses across chunk boundaries, separating a number from the sentence that explains it. Chunks of 400 words mixed multiple unrelated policy sections into one embedding, diluting the similarity signal for specific questions. 220/40 kept each chunk close to one coherent policy point while preserving surrounding context via overlap.
* Section headings (e.g., "5. Maternity and Paternity Leave") are auto-detected via regex and attached to each chunk when present, improving citation readability.

## 6\. Embedding Model

* **Model:** `all-MiniLM-L6-v2` (Sentence-Transformers), 384-dimensional embeddings, L2-normalized
* The **same model** embeds both document chunks and user queries — required, since embeddings from different models live in incompatible vector spaces and cannot be meaningfully compared
* Chosen for being fast, CPU-friendly, and strong on general-purpose semantic similarity; swappable for `all-mpnet-base-v2` for higher quality at the cost of latency

## 7\. Vector Database

* **FAISS** `IndexFlatIP` (exact inner-product search, equivalent to cosine similarity on normalized vectors)
* Chosen over a hosted vector DB (Pinecone/Weaviate/Qdrant) because it runs fully locally with no external service dependency or API cost, and at this document collection's scale (hundreds to low-thousands of chunks) exact search is both fast and easier to reason about than approximate search
* Persisted to `vectorstore/` as `index.faiss` + `chunks.pkl` (chunk metadata) + `meta.json` (embedding model name, dimension, chunk count) — supports reload without re-embedding

## 8\. Retrieval Strategy

* **Top-K retrieval**, K=4 by default (adjustable in the Streamlit sidebar, range 2–8)
* **Similarity threshold** (default 0.30 cosine similarity): chunks scoring below this are discarded *before* ever reaching the LLM. This is the first of two hallucination-handling layers (see Section 10).
* If zero chunks clear the threshold, the pipeline short-circuits and returns the "insufficient information" message without calling the LLM at all — saving cost and guaranteeing no hallucination in that case.

## 9\. Prompt Strategy

The system prompt (`src/rag\_pipeline.py::SYSTEM\_PROMPT`) explicitly instructs the model to:

1. Answer using **only** the provided context, never outside knowledge
2. Return an exact fallback string when the context is insufficient
3. Never fabricate figures/dates not literally present in the context
4. Stay concise and professional
5. Use conversation history only to interpret follow-up questions, not as a source of facts

Each retrieved chunk is injected into the prompt with an explicit `\[Source N: doc\_name, Page X — Section]` label, which both grounds the model's reasoning and lets us map its answer back to concrete citations for the UI.

## 10\. Hallucination Handling (two layers)

1. **Retrieval-level:** similarity-threshold filtering (Section 8) — weak/irrelevant chunks never reach the LLM.
2. **Prompt-level:** even when some retrieved content is only tangentially related, the system prompt instructs the model to output the exact fallback string — *"I could not find sufficient information in the provided documents to answer this question."* — rather than guessing. The pipeline checks for this exact string to set `grounded=False` and suppresses source citations in that case (no sources are shown for an answer the model didn't actually derive from the documents).

This two-layer design is deliberate: our evaluation (Section 12) showed that retrieval alone isn't perfectly selective on an out-of-scope question, but the prompt-level instruction still prevented a fabricated answer — see Limitations.

## 11\. LLM Used

**Anthropic Claude** via the official `anthropic` Python SDK (`src/rag\_pipeline.py::\_call\_llm`). Chosen for strong instruction-following on grounding constraints and easy swap-ability — the rest of the pipeline (retrieval, prompt construction, source attribution) is provider-agnostic, so switching to OpenAI, Gemini, or a local Hugging Face model only requires rewriting `\_call\_llm`.

Requires `GEMINI\_API\_KEY` set as an environment variable. **Never commit this key to GitHub** — it is excluded via `.gitignore`.

## 12\. Evaluation Methodology \& Results

`src/evaluation.py` defines an 8-question hand-labeled evaluation set (`EVAL\_DATASET`) covering all four documents plus one deliberately out-of-scope question ("Does the company provide a company car?") to test refusal behavior. For each question we record:

* Expected source document (or `None` for "should refuse")
* Reference answer (for manual grading)
* Retrieved documents (auto-computed)
* **Retrieval hit rate**: did the expected document appear among retrieved chunks?
* **Hallucination refusal accuracy**: for the out-of-scope question, did the system correctly refuse rather than fabricate an answer?

Run via `python src/evaluation.py` (needs a built index + `ANTHROPIC\_API\_KEY`) or `notebooks/03\_rag\_evaluation.ipynb`. Results are saved to `data/processed/evaluation\_results.json`.

**Sandbox validation:** the development sandbox used to build this project could not reach `huggingface.co` to download the real embedding model, so retrieval mechanics were validated with a TF-IDF stand-in (`scripts/test\_retrieval\_eval\_offline.py`) against the same evaluation questions:

|Metric|Result (TF-IDF stand-in)|
|-|-|
|Retrieval hit rate (7 answerable questions)|7/7 = 100%|
|Correct refusal (1 out-of-scope question)|0/1 — weak matches slipped past the retrieval threshold|

This is a useful, honest finding: real semantic embeddings (all-MiniLM-L6-v2) will score the out-of-scope question's cosine similarity much lower than TF-IDF's lexical overlap does, and the prompt-level refusal instruction (Section 10) is the second safety net specifically for cases like this where weak context slips through retrieval. **Re-run `python src/evaluation.py` in an environment with internet access to get final numbers with the real embedding model + live LLM.**

## 13\. Streamlit Deployment Instructions

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Anthropic API key
export ANTHROPIC\_API\_KEY=sk-ant-...        # macOS/Linux
setx ANTHROPIC\_API\_KEY "sk-ant-..."        # Windows

# 3. (One-time) build the vector index from data/documents/
python src/retriever.py

# 4. Launch the app
streamlit run app.py
```

Then, in the app: upload additional PDF/DOCX files via the sidebar (or use the bundled sample HR documents), click **"Build / Rebuild Index"**, and start asking questions in the chat box. Adjust Top-K and the similarity threshold in the sidebar to tune retrieval behavior live.

## 14\. Limitations

* Evaluation set is small (8 questions) — sufficient to demonstrate methodology, not a statistically robust benchmark
* `IndexFlatIP` exact search does not scale to millions of chunks — would need an approximate index (HNSW, IVF) or a hosted vector DB in production
* Retrieval threshold alone is not perfectly selective on out-of-scope questions (see Section 12); relies on the prompt-level instruction as a second safety net
* No reranking or query rewriting implemented (see Section 15 for what a production version would add)
* Single embedding model tested; a domain-tuned or larger model may improve retrieval precision further
* DOCX "page" numbers are synthetic (paragraph-group indices), not true page numbers, since DOCX has no native pagination

## 15\. Future Improvements

* Add a reranker (e.g., cross-encoder) over the top-20 retrieved chunks before truncating to top-K, to improve precision
* Query rewriting for ambiguous or multi-part questions
* Hybrid search (BM25 + vector) to catch exact keyword matches (e.g., specific policy clause numbers) that pure semantic search can miss
* RAGAS-based automated evaluation (faithfulness, answer relevance, context precision/recall) in place of the current hand-labeled set
* Metadata filtering (e.g., "only search the Leave Policy") exposed in the UI
* Streaming LLM responses for lower perceived latency
* Document re-indexing workflow that only re-embeds changed files instead of a full rebuild

## 16\. Technology Stack

Python · pypdf · python-docx · Sentence-Transformers (`all-MiniLM-L6-v2`) · FAISS · Anthropic Claude API · Streamlit · Git/GitHub

## 17\. Project Structure

```
enterprise-rag-assistant/
├── data/
│   ├── documents/              # source HR PDFs
│   └── processed/              # evaluation\_results.json (generated)
├── notebooks/
│   ├── 01\_document\_exploration.ipynb
│   ├── 02\_chunking\_and\_embeddings.ipynb
│   └── 03\_rag\_evaluation.ipynb
├── scripts/
│   ├── generate\_sample\_docs.py         # creates the sample HR document collection
│   ├── test\_pipeline\_offline.py        # sandbox-only smoke test (TF-IDF stand-in)
│   └── test\_retrieval\_eval\_offline.py  # sandbox-only eval harness validation
├── src/
│   ├── document\_loader.py      # Phase 1
│   ├── chunking.py             # Phase 2
│   ├── embeddings.py           # Phase 3
│   ├── retriever.py            # Phase 4 + 5
│   ├── rag\_pipeline.py         # Phase 6 + 7 + 8 + 9 + 10
│   └── evaluation.py           # Phase 18
├── vectorstore/                # FAISS index + metadata (generated)
├── app.py                      # Phase 19 — Streamlit application
├── requirements.txt
├── README.md
└── .gitignore
```

## 18\. Quick Reference: Key Concepts (for demonstration Q\&A)

* **RAG vs. asking an LLM directly:** RAG grounds answers in the organization's actual, current documents rather than the model's frozen training data — reducing hallucination and enabling source attribution and up-to-date answers without retraining.
* **Embedding:** a dense numeric vector representing the semantic meaning of text, positioned so semantically similar text lands nearby in vector space.
* **Cosine similarity:** measures the angle between two vectors (not magnitude), giving a similarity score independent of text length — used here via inner product on pre-normalized vectors.
* **Top-K retrieval:** retrieve the K chunks with highest similarity to the query embedding, rather than all chunks, to keep the LLM's context focused and relevant.

