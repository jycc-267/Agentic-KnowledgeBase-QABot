import logging
from langchain.schema import Document, HumanMessage, SystemMessage

from app import indexer
from app.core.llm import get_llm
from app.retrieval.common import SYSTEM_PROMPT, build_prompt, log_retrieval_event

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 5.0  # BM25 scores can be higher, so arbitrary threshold
RETRIEVAL_DEPTH = 5

def search(question: str, depth: int = RETRIEVAL_DEPTH) -> list[tuple[Document, float]]:
    return indexer.search_bm25(question, k=depth)

def query(question: str) -> dict:
    if indexer.bm25_index is None:
        return {
            "answer": "The knowledge base has not been indexed yet. Call POST /index first.",
            "sources": [],
        }

    bm25_results = search(question, RETRIEVAL_DEPTH)
    
    if not bm25_results:
        return {
            "answer": "I cannot confirm this from the knowledge base.",
            "sources": [],
        }

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

    top_k = bm25_results[:3]
    
    log_retrieval_event(
        mode="bm25",
        query=question,
        top_hits=[(doc.metadata.get("section_id", "unknown"), round(score, 4)) for doc, score in bm25_results],
    )

    response = get_llm().invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_prompt(question, top_k)),
    ])

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "heading": doc.metadata.get("heading", "unknown"),
            "score": round(float(score), 6),
            "content": doc.page_content[:240],
        }
        for doc, score in top_k
    ]

    return {
        "answer": response.content,
        "sources": sources,
    }
