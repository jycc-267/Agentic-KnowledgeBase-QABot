import logging
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app import indexer
from app.core.llm import get_llm
from app.retrieval.common import SYSTEM_PROMPT, build_prompt, log_retrieval_event

logger = logging.getLogger(__name__)

# FAISS distance is L2. Lower is better. We convert distance to similarity.
# Derived via tune_thresholds.py: midpoint of gap between
# lowest in-scope top-1 (0.5725) and highest OOS top-1 (0.5571).
CONFIDENCE_THRESHOLD = 0.5648
RETRIEVAL_DEPTH = 5

def search(question: str, depth: int = RETRIEVAL_DEPTH) -> list[tuple[Document, float]]:
    """
    Perform a semantic vector search against the FAISS index and resolve chunks to parent sections.
    
    Args:
        question: The user's query string.
        depth: The maximum number of top chunks to retrieve from FAISS.
        
    Returns:
        A list of (Document, similarity_score) tuples representing the top resolved sections,
        where similarity_score is higher for more relevant documents.
    """
    # Step 1: Execute the semantic search against the FAISS index (returns chunks and L2 distances)
    faiss_results = indexer.search_faiss(question, k=depth)
    
    # Step 2: Initialize structures for deduplicating parent sections and storing ranked results
    ranked: list[tuple[str, Document, float]] = []
    seen: set[str] = set()
    
    # Step 3: Iterate through each retrieved chunk to resolve its parent section
    for chunk_doc, distance in faiss_results:
        # Step 3a: Look up the full parent section Document using the chunk's metadata
        parent = indexer.resolve_parent_section(chunk_doc)
        sid = parent.metadata.get("section_id", chunk_doc.metadata.get("section_id", "unknown"))
        
        # Step 3b: Deduplicate, ensuring we only include a parent section once
        if sid not in seen:
            seen.add(sid)
            # Step 3c: Convert FAISS L2 distance (lower is better) to a similarity score (higher is better)
            similarity = 1.0 / (1.0 + float(distance))
            ranked.append((sid, parent, similarity))
    
    # Step 4: Strip out the section_id, leaving only the Document and its similarity score
    # Filter out results that fall below the confidence threshold
    results = [(doc, score) for _, doc, score in ranked if score >= CONFIDENCE_THRESHOLD]
    
    # Step 5: Log the retrieval event for observability
    log_retrieval_event(
        mode="vector",
        query=question,
        top_hits=[(doc.metadata.get("section_id", "unknown"), round(score, 4)) for doc, score in results],
    )
    
    # Step 6: Return the deduplicated and scored section Documents
    return results


def query(question: str) -> dict:
    """
    Main entry point for answering a user query using only Vector (semantic) retrieval.
    
    Args:
        question: The user's query string.
        
    Returns:
        A dictionary containing the generated 'answer' and a list of 'sources'.
    """
    # Step 1: Validate that the FAISS vector index has been initialized
    if indexer.vectorstore is None:
        return {
            "answer": "The knowledge base has not been indexed yet. Call POST /index first.",
            "sources": [],
        }

    # Step 2: Retrieve the best matching full section documents using semantic search
    ranked = search(question, RETRIEVAL_DEPTH)

    # Step 3: Handle the case where no documents were returned
    if not ranked:
        return {
            "answer": "I cannot confirm this from the knowledge base.",
            "sources": [],
        }

    # Step 4: Check if the top result meets the minimum confidence threshold
    top_score = ranked[0][1]
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
    top_k = ranked[:3]

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
