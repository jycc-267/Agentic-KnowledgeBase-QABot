# Implementation Plan: Hybrid Retrieval & Evaluation (BM25 + Vector RAG)

## Overview
This implementation plan maps the execution strategy for building a **Hybrid Retrieval System (BM25 + Vector RAG)** and a robust **Evaluation Framework** inside the `knowledge_base_qa_bot` repository. By combining exact keyword matching (BM25) with semantic search (FAISS) and introducing systematic evaluation for failure modes, we ensure highly reliable, source-citable answers that overcome the limitations of a single retrieval strategy.

## Design Logic & Architecture Q&A

### 1. Which retrieval strategy did you choose, and why?
- **Choice**: **Hybrid Retrieval (BM25 + Vector RAG) with Reciprocal Rank Fusion (RRF)**.
- **Reasoning**:
  - *Markdown KB (BM25)* is excellent at exact keyword matching and weighting rare terms (e.g., specific SKUs or names), but fails on synonyms or semantic rewrites (FM1: Keyword Miss).
  - *Vector RAG* excels at semantic understanding and paraphrasing, but can return tangentially related content that doesn't answer the intent (FM2: Semantic False Positive).
  - *Hybrid*: By running both and fusing their ranks (RRF), they compensate for each other's weaknesses.
  - *Cons*: Demands an external API provider/model to calculate embeddings (cost, latency, and dependency on Google Gemini APIs), plus larger dependency profiles (binary library compilers like `faiss-cpu`).
  - *Pros*: Vastly superior user experience, robust to query variations, and handles unstructured, natural language seamlessly.

### 2. What is the retrieval unit in your design: file, section, or chunk?
- **Choice**: **Dual Indexing with Parent-Section Mapping**.
- **Reasoning**:
  - *Sections* (Markdown KB) are great for providing complete context and clear citations, but can be too long and noisy for dense embeddings.
  - *Chunks* (Vector RAG) are optimal for vector search precision but suffer from fragmented context (FM3: Wrong Retrieval Unit).
  - *Solution*: We index sections for BM25 and chunks for FAISS. However, every chunk retains its parent section's metadata (`filename#heading-slug`). During retrieval, if a chunk hits, we resolve it back to its full parent section to feed the LLM, ensuring complete context and stable citation IDs.

### 3. How do you evaluate and handle Failure Modes?
- **Logic**: We implement an offline evaluation loop tracking **Recall@K**, **MRR (Mean Reciprocal Rank)**, and **Source Hit Rate**.
  - **FM1 (Keyword Miss)**: Solved by the semantic nature of Vector RAG.
  - **FM2 (Semantic False Positive)**: Mitigated by BM25 relevance and score thresholding.
  - **FM3 (Wrong Retrieval Unit)**: Solved by the "Parent-Section Mapping" described above.
  - **FM4 (Knowledge Gap)**: Solved by enforcing strict retrieval score thresholds and an explicit LLM fallback instruction ("I cannot confirm from the knowledge base.").

### 4. How do you decide what goes into the prompt?
- **Logic**:
  - *System Instruction Block*: Enforces strict grounding, forbids hallucination, defines the fallback protocol, and establishes citation syntax.
  - *Retrieved Context*: The top-$k$ fused and resolved parent sections. Each is prefixed with `[Source: filename#heading-slug]` to isolate sources.
  - *User Query*: Placed at the end to maximize attention alignment.

### 5. How do you cite sources so users can inspect the original Markdown?
- **Logic**: The resolved parent sections provide stable IDs (e.g., `refund_policy.md#refund-timeline`). The LLM is instructed to append these exact tags to facts. These IDs act as direct anchors to the canonical Markdown files for audit and UI deep-linking.

### 6. If the knowledge base grows from 10 files to 100,000 files, what changes?
- **Logic**:
  - Transition from in-memory FAISS to a scalable vector DB (pgvector, Pinecone).
  - Implement distributed ingestion (Spark, Celery) with incremental updates.
  - Introduce a Cross-Encoder Reranker stage to handle the increased semantic noise (Semantic False Positives) at scale.

## Codebase Context
- **Codebase Structure**: Built with **FastAPI** (`app/routes.py`, `app/main.py`), utilizing `app/indexer.py` and `app/retrieval.py`.
- **Target Integrations**: Requires adding a BM25 library (e.g., `rank_bm25`), implementing RRF logic, and building evaluation scripts/tables to run against an `eval_set.json`. Uses `langchain-google-genai` for Google/Gemini embeddings and LLM.

## Requirements
- **Hybrid Ingestion**: Parse docs into sections for BM25, and chunk those sections for FAISS.
- **Switchable Retrieval**: Implement a switchable version of retrieval strategies (BM25 only, Vector only, Hybrid) to allow for comparative analysis.
- **Hybrid Retrieval**: Query both indexes, normalize scores/ranks using RRF, and map chunk hits back to full sections.
- **Evaluation Framework**: Implement an eval runner to measure Recall@K and MRR based on a predefined set of queries and expected sources.
- **Audit & Safety**: Log retrieval events (scores, ranks, sources) for observability. Enforce strict fallback when thresholds aren't met.


---

## Implementation Steps

### Phase 1: Hybrid Ingestion & Parsing
1. **Implement Core Utilities** (`app/core/parser.py`, `app/core/llm.py`)
   - Action: Centralize LLM/Embedding initialization. Parse `.md` into full sections (for BM25). Split sections into chunks (~500 chars) tagged with parent section IDs.
2. **Build Dual Indexes** (`app/indexer.py`)
   - Action: Create an in-memory BM25 index for the sections. Build the FAISS index for the chunks. Serialize both to `.kb/`.

### Phase 2: Modular Switchable Retrieval & Fusion
3. **Implement Retrieval Strategies** (`app/retrieval/`)
   - Action: Break retrieval logic into distinct, modular strategies:
     - `bm25_search.py`: Queries only the section index.
     - `vector_search.py`: Queries the chunk index and resolves to sections.
     - `hybrid_search.py`: Queries both and ranks using Reciprocal Rank Fusion (RRF).
     - `common.py`: Shared prompt construction and score thresholding logic (triggers fallback if confidence is too low).
4. **Strategy Router** (`app/routes.py`, `app/retrieval/__init__.py`)
   - Action: Export a `STRATEGIES` registry to allow `/chat` endpoint to dynamically route queries based on the requested `mode` parameter.

### Phase 3: Comparative Evaluation & Observability
5. **Create Evaluation Set & Runner** (`scripts/run_eval.py` & `scripts/eval_set.json`)
   - Action: Define queries and expected sources. The script runs queries through all three retrieval strategies (BM25, Vector, Hybrid) and calculates comparative metrics (Recall@K, MRR, Hit Rate).
6. **Integrate Evaluation API** (`app/routes.py`)
   - Action: Expose a `GET /evaluate` endpoint that executes the `run_eval` logic and returns JSON metrics to the frontend.
7. **Implement Audit Logging** (`app/retrieval/common.py`)
   - Action: Log the active strategy, query, and top-k retrieved sources.

### Phase 4: Comparative User Interface (Browser UI)
8. **Build Web Frontend** (`app/static/index.html`)
   - Action: Create a lightweight Vanilla JS frontend with a minimalist Bjork theme containing:
     - **Chat Interface**: For standard single-mode queries.
     - **Strategy Switcher**: Dropdown to toggle between BM25, Vector, and Hybrid modes.
     - **Compare View**: Runs a single query against all three modes simultaneously in a 3-column layout.
     - **Evaluation Dashboard**: Fetches from `/evaluate` and displays a comparative metrics table.
9. **Serve Frontend** (`app/main.py` & `app/routes.py`)
   - Action: Configure FastAPI to mount the `app/static` directory and serve `index.html` at the root path (`/`).

## Testing Strategy
- **Evaluation Metrics**: Run the eval script. Target Recall@3 > 90% on the core intents.
- **Failure Mode Tests**: specifically test queries known to trigger FM1 (synonyms) and FM4 (out-of-scope) to ensure the hybrid model and thresholds handle them correctly.
