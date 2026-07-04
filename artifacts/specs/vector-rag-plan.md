# Implementation Plan: Vector RAG Completion

## Overview
This implementation plan maps the execution strategy for completing the **Vector RAG** strategy inside the `knowledge_base_qa_bot` repository. We will implement Markdown parsing, text splitting, FAISS index management, system prompt grounding, and integration routing to ensure reliable, source-citable client replies.

## Design Logic & Architecture Q&A

### 1. Which retrieval strategy did you choose, and why?
- **Choice**: **Strategy B: Vector RAG** (semantic retrieval using dense vector embeddings and an in-memory vector database).
- **Reasoning**:
  - *Semantic Understanding*: Traditional keyword search (like BM25 in Markdown KB) fails when queries use synonyms or paraphrases not matching the document text exactly (e.g., "delivery times" vs "shipping standard"). Vector embeddings capture deep semantic meanings.
  - *Contextual Alignment*: Dense embeddings mapped to FAISS allow for scalable mathematical similarity calculations that match the user's intent rather than purely literal characters.
- **Trade-offs**:
  - *Cons*: Demands an external API provider/model to calculate embeddings (cost, latency, and dependency on OpenAI), plus larger dependency profiles (binary library compilers like `faiss-cpu`).
  - *Pros*: Vastly superior user experience, robust to query variations, and handles unstructured, natural language seamlessly.

### 2. What is the retrieval unit in your design: file, section, or chunk?
- **Choice**: **Heading-level Sections split into size-constrained Chunks (approx. 500 characters)**.
- **Reasoning**:
  - Entire *files* are too large: they introduce background noise, quickly exceed model token limits, and inflate API costs.
  - Raw *heading-level sections* can be highly unequal in length. A very long section under a single header can dilute semantic focus when vectorized.
  - Resolving sections to *chunks of 500 characters* with semantic boundaries (`RecursiveCharacterTextSplitter`) ensures that each chunk holds a single, highly coherent concept. We retain parent context by preserving `filename` and `heading` attributes in document metadata.

### 3. How do you decide what goes into the prompt?
- **Logic**:
  - *System Instruction Block*: Enforces strict grounding, forbids hallucination or external knowledge leakage, and establishes standard citation syntax.
  - *Retrieved Context*: Ranks and includes the top-$k$ ($k=3$) relevant chunks. Each block is formatted with explicit source boundary markers: `[Source: filename#heading-slug]` to help the LLM isolate sources.
  - *User Query*: Placed at the very end to maximize attention alignment (recency effect).
- **Trade-offs**:
  - High $k$ increases comprehension but risks context dilution, cost, and latency.
  - Low $k$ (e.g., 1 or 2) is cheaper and faster but risks missing multi-document or distributed answers.
  - Setting $k=3$ balances this tradeoff optimally for a small customer-facing bot.

### 4. How do you cite sources so users can inspect the original Markdown?
- **Logic**:
  - Each chunk is ingested with rich metadata containing its source file (`filename`) and slugified section header (`heading-slug`).
  - In `build_prompt`, we attach `[Source: filename#heading-slug]` above each context chunk.
  - The `SYSTEM_PROMPT` instructs the LLM: *"You must cite your sources. For every fact or piece of information you provide, append the source identifier in the format [Source: filename#heading-slug]"*.
  - Since the source ID maps directly to standard Markdown anchors (e.g., `refund_policy.md#refund-timeline`), frontend interfaces can convert these tags into deep-linking HTML anchors (`<a href="docs/refund_policy.md#refund-timeline">`).

### 5. What should happen when retrieval finds weak or irrelevant results?
- **Logic**:
  - Return a strict fallback response: `"I cannot confirm from the knowledge base."`
  - In a vector-based system, similarity distance/score (e.g., L2 distance in FAISS) determines relevance. Chunks with distance scores above a specific threshold (e.g., FAISS L2 score $> 0.8$, or cosine similarity $< 0.7$, depending on scaling) are classified as out-of-scope.
  - Rather than forcing the LLM to make a guess or invent details (hallucination), we either drop the context before prompting or instruct the LLM via `SYSTEM_PROMPT` to respond with the exact fallback text if the retrieved context is unrelated to the user's question.

### 6. When would you switch from Markdown KB to Vector RAG?
- **Logic**:
  - When the knowledge base expands in vocabulary size (users start asking in diverse styles/synonyms).
  - When semantic search outclasses keyword matching (e.g., searching for "getting money back" needs to retrieve `refund_policy.md` even if the word "money" is never written in the file).
  - When operational resources permit hosting a small vector index (local or managed) and paying for embedding calls.

### 7. When would you switch from Vector RAG back to a Markdown index?
- **Logic**:
  - When the knowledge base consists of highly specific code identifiers, serial numbers, SKU numbers, or rigid terminology where semantic distance is a liability (causing false positives) and exact string matching is paramount.
  - When operational requirements mandate a completely offline/network-free local environment where calling an external embedding API is prohibited due to security rules, or where local embedded models are too heavy for low-powered target environments.
  - When system costs or API usage quotas must be kept strictly at zero.

### 8. If the knowledge base grows from 10 files to 100,000 files, what changes?
- **Logic**:
  1. **Storage/Database**: In-memory FAISS flat index scales linearly and will eventually exhaust RAM. We must transition to a dedicated, partitioned vector database (e.g., pgvector, Qdrant, Pinecone, or Milvus) with HNSW (Hierarchical Navigable Small World) indexing for $O(\log N)$ query speed.
  2. **Ingestion Speed**: Ingesting sequentially will take hours. We would need a distributed parallel worker pipeline (e.g., Celery, Spark) to parse, chunk, and embed documents in parallel, incorporating incremental updates (only re-indexing changed files using file hashes).
  3. **Retrieval Precision**: As document volume grows, the likelihood of "semantic pollution" (retrieving highly similar but contextually incorrect chunks) increases. We must implement a two-stage retrieval pipeline: retrieve top-$k$ ($k=50$) using dense vectors, and pass them through a cross-encoder **Reranker** model (e.g., Cohere Rerank, BGE-Reranker) to select the absolute top-$3$ chunks.

## Codebase Context
- **Codebase Structure**: Built with **FastAPI** (`app/routes.py` and `app/main.py`), using a logic layer for vector ingestion (`app/indexer.py`) and grounded querying (`app/retrieval.py`).
- **Codebase Features**: Currently has endpoints for GET `/health`, POST `/index`, and POST `/chat` with missing RAG logic stubs (TODOs).
- **Codebase Dependencies**: Leverages `faiss-cpu` for similarity search, `langchain` / `langchain-openai` for OpenAI embedding and LLM invocation, and `pydantic` for schema validation.

## Requirements
- Parse all `.md` files in `docs/` and chunk them at heading-level granularity.
- Preserve source metadata mapping to each document chunk in the format `filename#heading-slug`.
- Perform semantic splitting using `RecursiveCharacterTextSplitter` with a chunk size of 500 characters and semantic boundary separators.
- Construct and serialize a FAISS vector index under `.kb/faiss_index/` accompanied by an inspectable `metadata.json` file.
- Dynamically load the persisted FAISS index on application startup.
- Implement strict prompting rules using `gpt-4o-mini` to prevent hallucination, forcing a specific fallback if context is insufficient.
- Return structured API answers including citations mapping directly to source metadata.

## Architecture Changes
- **`app/indexer.py`**: Configure splitter, implement `load_markdown_sections`, implement FAISS `build_index`, implement `save_vector_index` and `load_vector_index`.
- **`app/retrieval.py`**: Configure grounded `SYSTEM_PROMPT` and `build_prompt` formatting context blocks with citation markers.

---

## Implementation Steps

### Phase 1: Markdown Document Ingestion & Chunking
1. **Configure Chunk Splitter** (File: [indexer.py](file:///Users/jimmychien/personal_project/build-moat-live-sessions/knowledge_base_qa_bot/app/indexer.py))
   - Action: Update `splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, separators=["\n\n", "\n", ". ", " "])`.
   - Why: Ensures semantic cohesion and guards against boundary truncation.
   - Dependencies: None
   - Risk: Low

2. **Implement Markdown Parser** (File: [indexer.py](file:///Users/jimmychien/personal_project/build-moat-live-sessions/knowledge_base_qa_bot/app/indexer.py))
   - Action: Implement `load_markdown_sections` to read files, split by `HEADING_RE`, capture sections, assign metadata `source` as `filename#slugified-heading`, and store actual header as `heading`.
   - Why: Enables source-level citations mapped directly to text sections.
   - Dependencies: Configure Chunk Splitter
   - Risk: Medium (needs resilient regex heading boundary matching)

### Phase 2: Index Building & Persistence
3. **Implement Index Builder** (File: [indexer.py](file:///Users/jimmychien/personal_project/build-moat-live-sessions/knowledge_base_qa_bot/app/indexer.py))
   - Action: Read files under `docs/`, parse sections, chunk with `splitter.split_documents()`, build FAISS store, and trigger save.
   - Why: Generates the vector database in memory.
   - Dependencies: Implement Markdown Parser
   - Risk: Low

4. **Implement Persistence Writers & Loaders** (File: [indexer.py](file:///Users/jimmychien/personal_project/build-moat-live-sessions/knowledge_base_qa_bot/app/indexer.py))
   - Action: Implement `save_vector_index` and `load_vector_index`. Serialize FAISS to local directory and write/read metadata metrics json.
   - Why: Guarantees warm-starts on server boot without repeating OpenAI api calls.
   - Dependencies: Implement Index Builder
   - Risk: Low

### Phase 3: Grounded Querying & Verification
5. **Formulate System Prompt** (File: [retrieval.py](file:///Users/jimmychien/personal_project/build-moat-live-sessions/knowledge_base_qa_bot/app/retrieval.py))
   - Action: Write explicit constraints inside `SYSTEM_PROMPT` to restrict responses to context and dictate citation formats and out-of-scope answers.
   - Why: Defends against hallucination and guarantees grounding.
   - Dependencies: None
   - Risk: Low

6. **Build Grounding Prompt** (File: [retrieval.py](file:///Users/jimmychien/personal_project/build-moat-live-sessions/knowledge_base_qa_bot/app/retrieval.py))
   - Action: Implement `build_prompt` to dynamically construct context blocks separated by their respective `[Source: filename#heading]` tags.
   - Why: Passes verified source metadata cleanly to the LLM.
   - Dependencies: Formulate System Prompt
   - Risk: Low

---

## Testing Strategy
- **Unit & Logic Tests**: Write mock assertions validating `load_markdown_sections` correctness.
- **Integration Tests**: Execute curl flows against endpoints `/health`, `/index`, and `/chat` using `uv run uvicorn`. Validate that out-of-scope replies triggerfallback and valid replies cite exact anchors.

## Risks & Mitigations
- **Risk**: Hallucination or leakage of general pre-trained knowledge. -> Mitigation: Formulate a highly restrictive prompt requiring the bot to refuse out-of-context answers with an exact phrase.
- **Risk**: API Rate-limits or Network failures during indexing. -> Mitigation: Persist local FAISS models immediately to avoid unnecessary model re-indexing.

## Success Criteria
- [ ] Correctly chunk documents based on Markdown headers.
- [ ] Generate local FAISS store and accompanying `metadata.json` under `.kb/faiss_index/`.
- [ ] Auto-load database on server reboot.
- [ ] Pass the out-of-scope questions with `"I cannot confirm from the knowledge base."`.
- [ ] Pass grounded questions citing accurate `filename#heading` anchors.

## Technical Debt (Flagged Red Flags)
- There is currently no unified logging setup; the application logs errors directly or uses `print`. We will preserve error handling standard structures using python's built-in `logging`.
