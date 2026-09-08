"""Hybrid Indexer: BM25 (sections) + FAISS (chunks) with parent-child mapping.

Design:
- Sections (full heading blocks) are indexed by BM25 for exact keyword retrieval.
- Chunks (~500 chars) are embedded into FAISS for semantic retrieval.
- Every chunk retains its parent section ID so FAISS hits resolve back to full
  sections, avoiding fragmented context (FM3: Wrong Retrieval Unit).
"""

import json
import logging
import os
import re
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from app.core.parser import load_markdown_sections, split_sections_to_chunks
from app.core.llm import get_embeddings, GEMINI_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
VECTOR_INDEX_DIR = Path(__file__).resolve().parents[1] / ".kb" / "faiss_index"
BM25_INDEX_DIR = Path(__file__).resolve().parents[1] / ".kb" / "bm25_index"

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
vectorstore: FAISS | None = None
bm25_index: BM25Okapi | None = None

# Section-level storage for BM25 retrieval and parent-resolution
_section_docs: list[Document] = []        # full section Documents
_section_id_to_doc: dict[str, Document] = {}  # section_id -> full section Doc

# Chunk-level storage for FAISS -> parent mapping
# (Note: LangChain FAISS documents retain metadata, so an external int->id map is not needed)

files_indexed = 0
sections_indexed = 0
chunks_indexed = 0


# ---------------------------------------------------------------------------
# Phase 1b: Dual index builder (BM25 + FAISS)
# ---------------------------------------------------------------------------

_NLTK_INITIALIZED = False

def _init_nltk():
    global _NLTK_INITIALIZED
    if _NLTK_INITIALIZED:
        return
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    _NLTK_INITIALIZED = True


def _tokenize(text: str) -> list[str]:
    """Tokenize using NLTK and remove stopwords.
    
    What: Converts raw text into a list of lowercase tokens with English stopwords removed.
    
    Why: Using a mature tokenizer and removing stopwords improves the signal-to-noise ratio
    for BM25 keyword matching, avoiding hits on common words like 'the' or 'is'.
    """
    _init_nltk()
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    return [t for t in tokens if t.isalnum() and t not in stop_words]


def build_index(docs_dir: Path = DOCS_DIR) -> tuple[int, int]:
    """Build dual indexes: BM25 over sections, FAISS over chunks.

    What:
    1. Loads and parses markdown into full sections.
    2. Builds a BM25 keyword index over the full sections.
    3. Splits those sections into smaller ~500 character chunks.
    4. Embeds those chunks into a FAISS vectorstore.
    5. Saves both indexes to disk for fast startup.

    Why: A hybrid approach combines the strengths of exact keyword matching (BM25 
    on large sections) with semantic meaning (FAISS on focused chunks). Building them
    together ensures our document IDs remain perfectly synchronized.
    
    Returns (files_indexed, sections_indexed).
    """
    global vectorstore, bm25_index, files_indexed, sections_indexed, chunks_indexed
    global _section_docs, _section_id_to_doc

    # Step 1: Discover and sort all markdown files in the docs directory
    md_files = sorted(docs_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No .md files found in {docs_dir}")

    # Step 2: Parse all discovered markdown files into full section-level Documents
    all_sections: list[Document] = []
    for md_file in md_files:
        all_sections.extend(load_markdown_sections(md_file))

    if not all_sections:
        raise ValueError("Parsing produced zero sections")

    # Step 3: Build the BM25 index over the full sections for keyword retrieval
    _section_docs = all_sections
    _section_id_to_doc = {
        doc.metadata["section_id"]: doc for doc in all_sections
    }
    tokenized_corpus = [_tokenize(doc.page_content) for doc in all_sections]
    bm25_index = BM25Okapi(tokenized_corpus)

    # Step 4: Split the large sections into smaller chunks for dense embedding
    all_chunks = split_sections_to_chunks(all_sections)

    # Step 5: Create the FAISS vectorstore by embedding the small chunks
    logger.info(f"Embedding {len(all_chunks)} chunks to FAISS using {GEMINI_EMBEDDING_MODEL}...")
    vectorstore = FAISS.from_documents(all_chunks, get_embeddings())
    logger.info("FAISS embedding complete.")

    # Step 6: Update global state counters for logging and persistence
    files_indexed = len(md_files)
    sections_indexed = len(all_sections)
    chunks_indexed = len(all_chunks)

    logger.info(
        "Built hybrid index: %d files, %d sections (BM25), %d chunks (FAISS)",
        files_indexed, sections_indexed, chunks_indexed,
    )

    # Step 7: Persist both indexes to disk for fast reload on next startup
    save_vector_index()
    _save_bm25_metadata()

    return files_indexed, sections_indexed


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_vector_index(index_dir: Path = VECTOR_INDEX_DIR) -> None:
    """Persist the FAISS index and metadata so restart doesn't re-embed.
    
    What: Writes the FAISS index and a metadata JSON to the .kb directory.
    
    Why: Generating embeddings costs time and API credits. Persisting them allows 
    subsequent bot startups to load instantly without hitting the embedding API again.
    """
    if vectorstore is None:
        return

    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))

    metadata = {
        "embedding_model": GEMINI_EMBEDDING_MODEL,
        "files_indexed": files_indexed,
        "sections_indexed": sections_indexed,
        "chunks_indexed": chunks_indexed,
    }
    (index_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    logger.info("FAISS index saved to %s", index_dir)


def _save_bm25_metadata(bm25_dir: Path = BM25_INDEX_DIR) -> None:
    """Persist BM25 section documents as JSON for reload on restart.
    
    What: Serializes the list of section Documents (content + metadata) to JSON
    using the new flat schema containing explicit tokens and heading paths.
    
    Why: Decouples index load from the tokenization logic and provides a strict
    schema that maps easily to production search engines.
    """
    if not _section_docs:
        return

    # Step 1: Ensure the output directory exists
    bm25_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 2: Transform the LangChain Documents into the flat JSON schema, explicitly tokenizing the content
    section_records = [
        {
            "id": doc.metadata.get("section_id"),
            "file": doc.metadata.get("source"),
            "heading_path": doc.metadata.get("heading_path", []),
            "content": doc.page_content,
            "tokens": _tokenize(doc.page_content)
        }
        for doc in _section_docs
    ]
    
    # Step 3: Serialize the records to sections.json
    (bm25_dir / "sections.json").write_text(
        json.dumps(section_records, indent=2), encoding="utf-8"
    )
    logger.info("BM25 section data saved to %s", bm25_dir)


def load_vector_index(index_dir: Path = VECTOR_INDEX_DIR) -> tuple[int, int]:
    """Load persisted FAISS + BM25 indexes on server startup.
    
    What: Checks for existing index files on disk. If found (and the embedding 
    model matches), it loads FAISS from disk and triggers BM25 reload.
    
    Why: Ensures rapid server starts for local development and production. The model
    check prevents crashes if the developer switches between OpenAI and Gemini embeddings.
    """
    global vectorstore, bm25_index, files_indexed, sections_indexed, chunks_indexed
    global _section_docs, _section_id_to_doc

    faiss_file = index_dir / "index.faiss"
    pkl_file = index_dir / "index.pkl"
    meta_file = index_dir / "metadata.json"

    # Step 1: Check if the required FAISS binary files exist
    if not (faiss_file.exists() and pkl_file.exists()):
        logger.info("No persisted FAISS index found at %s", index_dir)
        return 0, 0

    # Step 2: Check the metadata to ensure the embedding model matches the current configuration
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    if meta.get("embedding_model") != GEMINI_EMBEDDING_MODEL:
        logger.warning(
            "Embedding model mismatch (persisted=%s, current=%s) — skipping load",
            meta.get("embedding_model"), GEMINI_EMBEDDING_MODEL,
        )
        return 0, 0

    # Step 3: Safely deserialize and load the local FAISS index
    # FAISS stores Document objects in a local pickle. Only load indexes created by this app.
    vectorstore = FAISS.load_local(
        str(index_dir),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
    
    # Step 4: Restore global index counters
    files_indexed = meta.get("files_indexed", 0)
    sections_indexed = meta.get("sections_indexed", 0)
    chunks_indexed = meta.get("chunks_indexed", 0)

    # Step 5: Trigger the complementary load of the BM25 keyword index data
    _load_bm25_sections()

    logger.info(
        "Loaded persisted index: %d files, %d sections, %d chunks",
        files_indexed, sections_indexed, chunks_indexed,
    )
    return files_indexed, sections_indexed


def _load_bm25_sections(bm25_dir: Path = BM25_INDEX_DIR) -> None:
    """Reload BM25 index from persisted section data.
    
    What: Reads the JSON serialized sections, re-tokenizes them, and rebuilds BM25.
    
    Why: Restores the exact keyword index state that matches the FAISS chunks without
    having to rescan the original markdown files.
    """
    global bm25_index, _section_docs, _section_id_to_doc

    sections_file = bm25_dir / "sections.json"
    # Step 1: Return early if the serialized sections file is missing
    if not sections_file.exists():
        logger.info("No persisted BM25 sections found at %s", bm25_dir)
        return

    # Step 2: Parse the JSON records from disk
    records = json.loads(sections_file.read_text(encoding="utf-8"))
    
    _section_docs = []
    tokenized_corpus = []
    
    # Step 3: Iterate through records to reconstruct full LangChain Documents and extract tokens
    for r in records:
        doc = Document(
            page_content=r["content"], 
            metadata={
                "source": r["file"],
                "heading": r["heading_path"][-1] if r.get("heading_path") else "",
                "heading_path": r.get("heading_path", []),
                "section_id": r["id"]
            }
        )
        _section_docs.append(doc)
        tokenized_corpus.append(r.get("tokens", []))
        
    # Step 4: Rebuild the lookup dictionary for parent resolution
    _section_id_to_doc = {
        doc.metadata["section_id"]: doc for doc in _section_docs
    }
    
    # Step 5: Initialize the BM25Okapi index directly using the loaded explicit tokens
    bm25_index = BM25Okapi(tokenized_corpus)
    logger.info("BM25 index rebuilt from %d persisted sections", len(_section_docs))


# ---------------------------------------------------------------------------
# Search helpers (used by retrieval.py)
# ---------------------------------------------------------------------------

FAISS_MAX_RETRIES = 3
FAISS_BASE_DELAY = 1.0  # seconds


def search_faiss(query: str, k: int = 5) -> list[tuple[Document, float]]:
    """Semantic search via FAISS.  Returns (chunk_doc, distance) pairs.
    
    What: Converts the query to a vector and finds the `k` nearest chunks in FAISS.
    
    Why: Catches conceptual matches even if the exact keywords differ (e.g., "refund" 
    matching "money back"). Distance scores represent semantic similarity.
    
    Retry: The Gemini Embedding API enforces rate limits (429). On transient
    ResourceExhausted errors, retries up to FAISS_MAX_RETRIES times with
    exponential backoff (1s, 2s, 4s) before giving up.
    """
    if vectorstore is None:
        return []

    last_error: Exception | None = None
    for attempt in range(FAISS_MAX_RETRIES + 1):
        try:
            return vectorstore.similarity_search_with_score(query, k=k)
        except Exception as exc:
            error_str = str(exc).lower()
            is_rate_limit = "429" in error_str or "resource exhausted" in error_str
            if not is_rate_limit or attempt >= FAISS_MAX_RETRIES:
                raise
            last_error = exc
            delay = FAISS_BASE_DELAY * (2 ** attempt)
            logger.warning(
                "Embedding API rate-limited (attempt %d/%d), retrying in %.1fs...",
                attempt + 1, FAISS_MAX_RETRIES, delay,
            )
            time.sleep(delay)

    # Should not reach here, but satisfy type checker
    if last_error:
        raise last_error
    return []


def search_bm25(query: str, k: int = 5) -> list[tuple[Document, float]]:
    """Keyword search via BM25.  Returns (section_doc, bm25_score) pairs.
    
    What: Tokenizes the query and returns the `k` top scoring sections using BM25.
    
    Why: Catches exact product names, IDs, or specific terminology that might be
    missed by dense semantic embeddings.
    """
    if bm25_index is None or not _section_docs:
        return []
    tokens = _tokenize(query)
    scores = bm25_index.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
    return [(_section_docs[idx], float(score)) for idx, score in ranked if score > 0]


def resolve_parent_section(doc: Document) -> Document:
    """Given a chunk Document, return its full parent section Document.

    What: Looks up the `parent_section_id` in the `_section_id_to_doc` dictionary.
    If the doc is already a section or the parent cannot be found, it returns the 
    original doc unchanged.
    
    Why: To avoid providing fragmented, cut-off sentences to the LLM (FM3), we retrieve
    small chunks for high semantic precision, but expand them to their full parent 
    heading section before synthesizing the final answer.
    """
    parent_id = doc.metadata.get("parent_section_id")
    if parent_id and parent_id in _section_id_to_doc:
        return _section_id_to_doc[parent_id]
    return doc
