import logging
from collections import defaultdict
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app import indexer
from app.core.llm import get_llm
from app.retrieval.common import SYSTEM_PROMPT, build_prompt, log_retrieval_event

logger = logging.getLogger(__name__)

# Derived via tune_thresholds.py grid sweep:
# - RRF_K=10 weights top-ranked results more aggressively (composite=0.9781)
# - FAISS_THRESHOLD=0.68 filters noisy semantic matches before RRF fusion
# - CONFIDENCE_THRESHOLD=0.005 stays low since OOS is handled by FAISS_THRESHOLD
RRF_K = 10
CONFIDENCE_THRESHOLD = 0.005
RETRIEVAL_DEPTH = 5
FAISS_THRESHOLD = 0.68

def _reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, Document]]],
    k: int = RRF_K,
) -> list[tuple[str, Document, float]]:
    """
    Fuse multiple ranked lists into a single ranked list using Reciprocal Rank Fusion (RRF).
    
    RRF assigns a score to each item based on its rank (position) in the input lists,
    rather than its raw score (which may not be comparable across different search engines).
    
    Args:
        ranked_lists: A list of ranked lists, where each inner list contains (section_id, Document) tuples.
        k: A smoothing constant to mitigate the impact of high-ranking outliers.
        
    Returns:
        A single list of (section_id, Document, rrf_score) tuples, sorted by rrf_score in descending order.
    """
    # Step 1: Initialize dictionaries to accumulate RRF scores and store unique Documents
    scores: dict[str, float] = defaultdict(float)
    docs: dict[str, Document] = {}

    # Step 2: Iterate over each ranked list (e.g., BM25 list, FAISS list)
    for ranked in ranked_lists:
        # Step 3: Iterate through the items in the ranked list, tracking their rank (1-indexed)
        for rank, (section_id, doc) in enumerate(ranked, start=1):
            # Step 4: Add the RRF positional penalty score for this item
            scores[section_id] += 1.0 / (k + rank)
            # Step 5: Save the document object if we haven't seen it yet
            if section_id not in docs:
                docs[section_id] = doc

    # Step 6: Combine the accumulated scores and documents into a flat list
    fused = [
        (sid, docs[sid], score)
        for sid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]
    
    # Step 7: Return the fused list, which is already sorted by the highest RRF score
    return fused

def search(
    question: str,
    depth: int = RETRIEVAL_DEPTH,
) -> list[tuple[Document, float]]:
    """
    Perform a hybrid search combining BM25 (keyword) and FAISS (semantic) retrieval.
    
    Args:
        question: The user's query string.
        depth: The number of top results to retrieve from each individual search engine.
        
    Returns:
        A list of (Document, score) tuples representing the top combined results.
    """
    # Step 1: Execute exact-keyword search using BM25
    bm25_results = indexer.search_bm25(question, k=depth)
    
    # Step 2: Format BM25 results into a list of (section_id, Document) tuples for fusion
    bm25_ranked: list[tuple[str, Document]] = [
        (doc.metadata["section_id"], doc) for doc, _score in bm25_results
    ]

    # Step 3: Execute semantic vector search using FAISS
    faiss_results = indexer.search_faiss(question, k=depth)
    
    faiss_ranked: list[tuple[str, Document]] = []
    seen_faiss_sections: set[str] = set()
    
    # Step 4: Resolve each retrieved FAISS chunk back to its full parent section
    for chunk_doc, distance in faiss_results:
        # Step 4a: Filter out low-confidence semantic matches
        similarity = 1.0 / (1.0 + float(distance))
        if similarity < FAISS_THRESHOLD:
            continue
            
        # Step 4b: Look up the full parent section document using the chunk's metadata
        parent = indexer.resolve_parent_section(chunk_doc)
        sid = parent.metadata.get("section_id", chunk_doc.metadata.get("section_id", "unknown"))
        
        # Step 4c: Deduplicate parent sections (multiple chunks might belong to the same section)
        if sid not in seen_faiss_sections:
            seen_faiss_sections.add(sid)
            faiss_ranked.append((sid, parent))

    # Step 5: Fuse the BM25 and resolved FAISS lists together using RRF
    fused = _reciprocal_rank_fusion([bm25_ranked, faiss_ranked])
    
    # Step 6: Log the retrieval event for observability and debugging
    log_retrieval_event(
        mode="hybrid",
        query=question,
        top_hits=[(sid, round(score, 6)) for sid, _, score in fused],
    )

    # Step 7: Strip out the section_id and return only the Document and its RRF score
    return [(doc, score) for _sid, doc, score in fused]

def query(question: str) -> dict:
    """
    Main entry point for answering a user query using the hybrid RAG architecture.
    
    Args:
        question: The user's query string.
        
    Returns:
        A dictionary containing the generated 'answer' and a list of 'sources'.
    """
    # Step 1: Validate that the knowledge base indices have been initialized
    if indexer.vectorstore is None and indexer.bm25_index is None:
        return {
            "answer": "The knowledge base has not been indexed yet. Call POST /index first.",
            "sources": [],
        }

    # Step 2: Retrieve the best matching documents using the hybrid search pipeline
    ranked_sections = search(question)

    # Step 3: Handle the case where no documents were returned
    if not ranked_sections:
        return {
            "answer": "I cannot confirm this from the knowledge base.",
            "sources": [],
        }

    # Step 4: Check if the top result meets the minimum confidence threshold
    top_score = ranked_sections[0][1]
    if top_score < CONFIDENCE_THRESHOLD:
        logger.info(
            "Below confidence threshold (%.6f < %.6f) — triggering fallback",
            top_score, CONFIDENCE_THRESHOLD,
        )
        return {
            "answer": "I cannot confirm this from the knowledge base.",
            "sources": [],
        }

    # Step 5: Extract the top K documents to send to the LLM (e.g., top 3)
    top_k = ranked_sections[:3]

    # Step 6: Invoke the LLM with the system prompt and the formatted retrieved context
    response = get_llm().invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_prompt(question, top_k)),
    ])

    # Step 7: Format the sources list to include metadata and a snippet of the content
    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "heading": doc.metadata.get("heading", "unknown"),
            "score": round(float(score), 6),
            "content": doc.page_content[:240],
        }
        for doc, score in top_k
    ]

    # Step 8: Return the final synthesized answer and its citations
    return {
        "answer": response.content,
        "sources": sources,
    }
