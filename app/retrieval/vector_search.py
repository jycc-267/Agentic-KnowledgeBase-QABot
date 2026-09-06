import logging
from langchain.schema import Document, HumanMessage, SystemMessage

from app import indexer
from app.core.llm import get_llm
from app.retrieval.common import SYSTEM_PROMPT, build_prompt, log_retrieval_event

logger = logging.getLogger(__name__)

# FAISS distance is L2. Lower is better. We convert distance to similarity.
CONFIDENCE_THRESHOLD = 0.5
RETRIEVAL_DEPTH = 5

def search(question: str, depth: int = RETRIEVAL_DEPTH) -> list[tuple[Document, float]]:
    faiss_results = indexer.search_faiss(question, k=depth)
    
    ranked: list[tuple[str, Document, float]] = []
    seen: set[str] = set()
    for chunk_doc, distance in faiss_results:
        parent = indexer.resolve_parent_section(chunk_doc)
        sid = parent.metadata.get("section_id", chunk_doc.metadata.get("section_id", "unknown"))
        if sid not in seen:
            seen.add(sid)
            similarity = 1.0 / (1.0 + float(distance))
            ranked.append((sid, parent, similarity))
    
    return [(doc, score) for _, doc, score in ranked]


def query(question: str) -> dict:
    if indexer.vectorstore is None:
        return {
            "answer": "The knowledge base has not been indexed yet. Call POST /index first.",
            "sources": [],
        }

    ranked = search(question, RETRIEVAL_DEPTH)

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

    top_k = ranked[:3]
    
    log_retrieval_event(
        mode="vector",
        query=question,
        top_hits=[(doc.metadata.get("section_id", "unknown"), round(score, 4)) for doc, score in ranked],
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
