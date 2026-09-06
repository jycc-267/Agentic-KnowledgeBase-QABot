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
from pathlib import Path

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

from app.core.parser import load_markdown_sections, split_sections_to_chunks
from app.core.llm import get_embeddings, GEMINI_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
INDEX_DIR = Path(__file__).resolve().parents[1] / ".kb" / "faiss_index"
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
_chunk_to_section_id: dict[int, str] = {}  # chunk index -> parent section_id

files_indexed = 0
sections_indexed = 0
chunks_indexed = 0


chunks_indexed = 0


# ---------------------------------------------------------------------------
# Phase 1b: Dual index builder (BM25 + FAISS)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_index(docs_dir: Path = DOCS_DIR) -> tuple[int, int]:
    """Build dual indexes: BM25 over sections, FAISS over chunks.

    Returns (files_indexed, sections_indexed).
    """
    global vectorstore, bm25_index, files_indexed, sections_indexed, chunks_indexed
    global _section_docs, _section_id_to_doc, _chunk_to_section_id

    md_files = sorted(docs_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No .md files found in {docs_dir}")

    # 1. Parse all files into sections
    all_sections: list[Document] = []
    for md_file in md_files:
        all_sections.extend(load_markdown_sections(md_file))

    if not all_sections:
        raise ValueError("Parsing produced zero sections")

    # 2. Build BM25 index over full sections
    _section_docs = all_sections
    _section_id_to_doc = {
        doc.metadata["section_id"]: doc for doc in all_sections
    }
    tokenized_corpus = [_tokenize(doc.page_content) for doc in all_sections]
    bm25_index = BM25Okapi(tokenized_corpus)

    # 3. Split sections into chunks for FAISS
    all_chunks = split_sections_to_chunks(all_sections)

    # 4. Build chunk-index -> parent section_id mapping
    _chunk_to_section_id = {
        i: chunk.metadata["parent_section_id"]
        for i, chunk in enumerate(all_chunks)
    }

    # 5. Create FAISS vectorstore from chunks
    vectorstore = FAISS.from_documents(all_chunks, get_embeddings())

    files_indexed = len(md_files)
    sections_indexed = len(all_sections)
    chunks_indexed = len(all_chunks)

    logger.info(
        "Built hybrid index: %d files, %d sections (BM25), %d chunks (FAISS)",
        files_indexed, sections_indexed, chunks_indexed,
    )

    # 6. Persist indexes
    save_vector_index()
    _save_bm25_metadata()

    return files_indexed, sections_indexed


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_vector_index(index_dir: Path = INDEX_DIR) -> None:
    """Persist the FAISS index and metadata so restart doesn't re-embed."""
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
    """Persist BM25 section documents as JSON for reload on restart."""
    if not _section_docs:
        return

    bm25_dir.mkdir(parents=True, exist_ok=True)
    section_records = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in _section_docs
    ]
    (bm25_dir / "sections.json").write_text(
        json.dumps(section_records, indent=2), encoding="utf-8"
    )
    logger.info("BM25 section data saved to %s", bm25_dir)


def load_vector_index(index_dir: Path = INDEX_DIR) -> tuple[int, int]:
    """Load persisted FAISS + BM25 indexes on server startup."""
    global vectorstore, bm25_index, files_indexed, sections_indexed, chunks_indexed
    global _section_docs, _section_id_to_doc

    faiss_file = index_dir / "index.faiss"
    pkl_file = index_dir / "index.pkl"
    meta_file = index_dir / "metadata.json"

    if not (faiss_file.exists() and pkl_file.exists()):
        logger.info("No persisted FAISS index found at %s", index_dir)
        return 0, 0

    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    if meta.get("embedding_model") != GEMINI_EMBEDDING_MODEL:
        logger.warning(
            "Embedding model mismatch (persisted=%s, current=%s) — skipping load",
            meta.get("embedding_model"), GEMINI_EMBEDDING_MODEL,
        )
        return 0, 0

    vectorstore = FAISS.load_local(
        str(index_dir),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
    files_indexed = meta.get("files_indexed", 0)
    sections_indexed = meta.get("sections_indexed", 0)
    chunks_indexed = meta.get("chunks_indexed", 0)

    # Reload BM25 section data
    _load_bm25_sections()

    logger.info(
        "Loaded persisted index: %d files, %d sections, %d chunks",
        files_indexed, sections_indexed, chunks_indexed,
    )
    return files_indexed, sections_indexed


def _load_bm25_sections(bm25_dir: Path = BM25_INDEX_DIR) -> None:
    """Reload BM25 index from persisted section data."""
    global bm25_index, _section_docs, _section_id_to_doc

    sections_file = bm25_dir / "sections.json"
    if not sections_file.exists():
        logger.info("No persisted BM25 sections found at %s", bm25_dir)
        return

    records = json.loads(sections_file.read_text(encoding="utf-8"))
    _section_docs = [
        Document(page_content=r["page_content"], metadata=r["metadata"])
        for r in records
    ]
    _section_id_to_doc = {
        doc.metadata["section_id"]: doc for doc in _section_docs
    }
    tokenized_corpus = [_tokenize(doc.page_content) for doc in _section_docs]
    bm25_index = BM25Okapi(tokenized_corpus)
    logger.info("BM25 index rebuilt from %d persisted sections", len(_section_docs))


# ---------------------------------------------------------------------------
# Search helpers (used by retrieval.py)
# ---------------------------------------------------------------------------

def search_faiss(query: str, k: int = 5) -> list[tuple[Document, float]]:
    """Semantic search via FAISS.  Returns (chunk_doc, distance) pairs."""
    if vectorstore is None:
        return []
    return vectorstore.similarity_search_with_score(query, k=k)


def search_bm25(query: str, k: int = 5) -> list[tuple[Document, float]]:
    """Keyword search via BM25.  Returns (section_doc, bm25_score) pairs."""
    if bm25_index is None or not _section_docs:
        return []
    tokens = _tokenize(query)
    scores = bm25_index.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
    return [(_section_docs[idx], float(score)) for idx, score in ranked if score > 0]


def resolve_parent_section(doc: Document) -> Document:
    """Given a chunk Document, return its full parent section Document.

    If the doc is already a section (no parent_section_id) or the parent
    cannot be found, return the original doc unchanged.
    """
    parent_id = doc.metadata.get("parent_section_id")
    if parent_id and parent_id in _section_id_to_doc:
        return _section_id_to_doc[parent_id]
    return doc


# Legacy search alias (backward compatibility with routes.py)
def search(query: str, k: int = 3) -> list[tuple[Document, float]]:
    """Backward-compatible FAISS-only search."""
    return search_faiss(query, k=k)
