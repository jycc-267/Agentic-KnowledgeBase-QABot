import logging
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app import indexer
from app.core.llm import get_llm
from app.retrieval.common import SYSTEM_PROMPT, build_prompt, log_retrieval_event

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.7544
RETRIEVAL_DEPTH = 5

def search(question: str, depth: int = RETRIEVAL_DEPTH) -> list[tuple[Document, float]]:
    """
    Perform an exact-keyword search against the BM25 index.
    
    Args:
        question: The user's query string.
        depth: The maximum number of top results to retrieve.
        
    Returns:
        A list of (Document, score) tuples representing the top keyword matches.
    """
    # Step 1: Execute the search against the BM25 index built in indexer.py
    results = indexer.search_bm25(question, k=depth)
    
    # Step 2: Log the retrieval event with the section IDs and their BM25 scores
    log_retrieval_event(
        mode="bm25",
        query=question,
        top_hits=[(doc.metadata.get("section_id", "unknown"), round(score, 4)) for doc, score in results],
    )
    
    # Step 3: Return the scored documents
    return results

def query(question: str) -> dict:
    """
    Main entry point for answering a user query using only BM25 (keyword) retrieval.
    
    Args:
        question: The user's query string.
        
    Returns:
        A dictionary containing the generated 'answer' and a list of 'sources'.
    """
    # Step 1: Validate that the BM25 knowledge base index has been initialized
    if indexer.bm25_index is None:
        return {
            "answer": "The knowledge base has not been indexed yet. Call POST /index first.",
            "sources": [],
        }

    # Step 2: Retrieve the best matching full section documents using BM25
    bm25_results = search(question, RETRIEVAL_DEPTH)
    
    # Step 3: Handle the case where no documents were returned
    if not bm25_results:
        return {
            "answer": "I cannot confirm this from the knowledge base.",
            "sources": [],
        }

    # Step 4: Check if the top result meets the minimum confidence threshold
    top_score = bm25_results[0][1]
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
    top_k = bm25_results[:3]

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
