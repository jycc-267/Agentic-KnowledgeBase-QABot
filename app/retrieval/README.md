# Indexing, Chunking, Retrieval, and Post-Retrieval

**1. Index Definition and Data Structures**

* How is an "index" defined in this specific architecture, and how does it differ from simply tokenizing or splitting text into paragraphs?
* What are the functional and structural differences between a sparse index and a dense index?
* *Requirement:* Please provide concrete JSON or data structure examples illustrating what the index looks like for both sparse and dense representations.

**2. Chunking Strategy and Parent-Child Resolution**

* What is the architectural rationale for implementing parent-child resolution during indexing?
* What is the exact structural difference between a "section" and a "chunk"?
* By what logic or algorithm are sections split down into smaller chunks?
* *Requirement:* Please provide a concrete data structure example showing the relationship between a parent section and its child chunks.

**3. Hybrid Retrieval Mechanics**

* Does the "hybrid approach" refer strictly to the retrieval phase, or does it also impact how the text is indexed?
* How are these indices actually queried and retrieved in practice?
* *Requirement:* Please map out the step-by-step retrieval process demonstrating exactly how a query extracts data from both indices.

**4. Post-Retrieval Pipeline: RRF and Reranking**

* **Reciprocal Rank Fusion (RRF):** Why is RRF necessary in a hybrid approach? What is the mathematical or logical function of RRF, and at what specific stage in the retrieval pipeline does it take place?
* **Reranking:** Why is a separate reranking step required after the initial hybrid retrieval? What is the reranker actually doing (e.g., cross-encoder scoring), and where does this sit in relation to RRF?

---

## 1. Index Definition and Data Structures

In this architecture, an **index** is an optimized data structure designed for fast retrieval of documents based on a query. It goes far beyond simply tokenizing text; it involves structuring the data so search algorithms (like exact keyword match or semantic similarity) can efficiently query it.

*   **Sparse Index (BM25):** Optimized for **exact keyword matching**. It operates on full heading-level "Sections". It uses Term Frequency-Inverse Document Frequency (TF-IDF) principles, mapping specific keywords to the documents they appear in. It's called "sparse" because most documents contain only a tiny fraction of the total vocabulary, resulting in mostly empty (zero) vectors.
*   **Dense Index (FAISS):** Optimized for **semantic meaning**. It operates on smaller text "Chunks". It converts text into dense, high-dimensional floating-point vectors using an embedding model (`GEMINI_EMBEDDING_MODEL`). It is "dense" because every dimension of the vector has a non-zero, continuous value representing abstract semantic concepts.

### Concrete Data Structures

**Sparse Index Record (BM25 persisted in `sections.json`)**
The sparse index stores full sections and their explicit tokenization:
```json
{
  "id": "refund-policy.md#refund-timeline",
  "file": "refund-policy.md",
  "heading_path": [
    "Refund Policy",
    "Refund Timeline"
  ],
  "content": "Refunds are processed within 5-7 business days after approval. The funds will be returned to the original payment method. If you used a credit card, please allow an additional billing cycle for the credit to appear.",
  "tokens": [
    "refunds",
    "processed",
    "within",
    "5-7",
    "business",
    "days",
    "approval",
    "funds",
    "returned",
    "original",
    "payment",
    "method"
  ]
}
```

**Dense Index Representation (FAISS Chunk in memory/pickle)**
The dense index maps high-dimensional vectors to a LangChain `Document`. Notice the `parent_section_id` in the metadata, which links back to the sparse index `id`:
```json
{
  "page_content": "Refunds are processed within 5-7 business days after approval. The funds will be returned to the original payment method.",
  "metadata": {
    "source": "refund-policy.md",
    "heading": "Refund Timeline",
    "heading_path": [
      "Refund Policy",
      "Refund Timeline"
    ],
    "section_id": "refund-policy.md#refund-timeline",
    "parent_section_id": "refund-policy.md#refund-timeline"
  }
}
// Note: The actual FAISS index stores the 768-dimensional float vector 
// [0.0123, -0.4521, 0.9912, ...] mapped to this LangChain Document object. 
// This mapping is handled internally by the LangChain vector store (`FAISS.from_documents()`) 
// which stores the serialized Document in its `docstore`. During retrieval, the FAISS 
// wrapper returns the Document object directly, meaning our retrieval code operates 
// purely on Documents rather than raw vectors.
```

---

## 2. Chunking Strategy and Parent-Child Resolution

### Architectural Rationale
The architecture explicitly implements **parent-child resolution** to solve the **Wrong Retrieval Unit (FM3)** failure mode—also known as context fragmentation. 
*   **Embeddings (Dense Search)** work best on short, focused snippets of text (~500 chars). If you embed a massive page, the semantic signal gets diluted.
*   **LLMs (Generation)** work best with full, coherent context. If you feed an LLM an isolated 500-char chunk that starts mid-sentence or lacks the preceding heading definition, it often hallucinates or misinterprets the text.

Parent-child resolution solves this: retrieve via the highly precise small chunk, but inflate that chunk to its full parent heading section before passing it to the LLM. This process explicitly takes place in [`app/indexer.py`](../indexer.py) inside the `resolve_parent_section(doc: Document)` function, which does an O(1) lookup in the `_section_id_to_doc` dictionary.

### Sections vs. Chunks
*   **Section:** A logical block of text defined by the Markdown structure (e.g., everything under an `###` heading). It is structurally coherent and can be large.
*   **Chunk:** An arbitrary, character-length subdivision of a Section. It is represented as a LangChain `Document` object. Sections are chopped down via the `RecursiveCharacterTextSplitter` (max 500 chars, 50 char overlap) preferring natural boundaries (`\n\n`, `\n`, `.`).

### Concrete Data Structure: Parent-Child Relationship

```python
# The Parent SECTION (Indexed in BM25)
Document(
    page_content="Refunds take 5-7 days. \n\n International refunds take 10-14 days. \n\n Expedited shipping is not refunded.",
    metadata={
        "source": "refunds.md",
        "heading": "Timelines",
        "heading_path": ["Refunds", "Timelines"],
        "section_id": "refunds.md#timelines"
    }
)

# The Child CHUNKS (Indexed in FAISS)
# Chunk 1
Document(
    page_content="Refunds take 5-7 days. \n\n International refunds take 10-14 days.",
    metadata={
        "source": "refunds.md",
        "heading": "Timelines",
        "heading_path": ["Refunds", "Timelines"],
        "section_id": "refunds.md#timelines",
        "parent_section_id": "refunds.md#timelines" # Points back to Parent
    }
)

# Chunk 2 (Note the 50 char overlap)
Document(
    page_content="International refunds take 10-14 days. \n\n Expedited shipping is not refunded.",
    metadata={
        "source": "refunds.md",
        "heading": "Timelines",
        "heading_path": ["Refunds", "Timelines"],
        "section_id": "refunds.md#timelines",
        "parent_section_id": "refunds.md#timelines" # Points back to Parent
    }
)
```

---

## 3. Hybrid Retrieval Mechanics

The "hybrid approach" fundamentally impacts **both indexing and retrieval**. 
*   **At Indexing:** The system builds *two separate indices* side-by-side (`indexer.py`): BM25 for Sections and FAISS for Chunks.
*   **At Retrieval:** The system queries both indices simultaneously.

### Step-by-Step Retrieval Process (`hybrid_search.py`)

1.  **Parallel Query Execution:** The user's query (e.g., "international return") is sent to both `search_bm25()` and `search_faiss()` concurrently.
2.  **BM25 Retrieval:** The BM25 engine tokenizes the query and returns the Top-K (e.g., 5) full **Sections** that contain exact keyword matches.
3.  **FAISS Retrieval:** The FAISS engine converts the query to a vector, calculates semantic similarity (L2 distance), and returns the Top-K most semantically similar **Chunks**.
4.  **Parent Resolution (The Bridge):** For each Chunk returned by FAISS, the system looks up its `parent_section_id` in the `_section_id_to_doc` dictionary and replaces the Chunk with its full parent **Section**.
5.  **Deduplication:** Multiple small chunks retrieved by FAISS might belong to the same parent section. These duplicates are stripped out so the section is only represented once.
6.  **Fusion (RRF):** The BM25 list (Sections) and the FAISS list (Resolved Sections) are merged into a single ranked list using Reciprocal Rank Fusion.
7.  **LLM Prompting:** The top 3 sections from the final fused list are injected into the prompt and sent to the LLM to generate the answer.

---

## 4. Post-Retrieval Pipeline: RRF and Reranking

### Reciprocal Rank Fusion (RRF)
**Why is it necessary?** 
BM25 and Vector Search use completely different scoring paradigms. BM25 outputs unbounded TF-IDF scores (e.g., `14.5`, `3.2`), while FAISS outputs distance metrics (e.g., L2 distance `0.4`, `0.9`). You **cannot mathematically compare or add these raw scores together**.

**Mathematical Function:**
Instead of raw scores, RRF looks at the document's *Rank* (position 1, 2, 3...) in each respective list. It assigns a new score based on the formula: 
`Score = 1 / (k + rank)` *(where k is a constant, set to 60 in this codebase)*.

If a document ranks 1st in BM25 and 3rd in FAISS, its RRF score is:
`1 / (60 + 1) + 1 / (60 + 3) = 0.01639 + 0.01587 = 0.03226`

**Pipeline Stage:** RRF happens directly in `hybrid_search.py` *after* the parallel retrieval and parent resolution, but *before* applying the `CONFIDENCE_THRESHOLD` or sending context to the LLM.

### Reranking
In the current architectural implementation of this specific repository, **there is no separate, dedicated reranker model** (such as a Cross-Encoder like `bge-reranker` or `Cohere Rerank`). 

In this system, **RRF acts as the reranking mechanism**. It reorders the combined lists based on their overlapping ranks. 

If a dedicated Cross-Encoder Reranker were to be added in the future, its purpose would be to take the Top ~20 documents output by RRF and feed them (along with the user query) into a heavier neural network to calculate a highly accurate relevance score. It would sit **after RRF, but before the LLM generation phase.**
