import logging
from collections import defaultdict
from langchain.schema import Document, HumanMessage, SystemMessage

from app import indexer
from app.core.llm import get_llm
from app.retrieval.common import SYSTEM_PROMPT, build_prompt, log_retrieval_event

logger = logging.getLogger(__name__)

RRF_K = 60
CONFIDENCE_THRESHOLD = 0.005
RETRIEVAL_DEPTH = 5

def _reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, Document]]],
    k: int = RRF_K,
) -> list[tuple[str, Document, float]]:
    scores: dict[str, float] = defaultdict(float)
    docs: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, (section_id, doc) in enumerate(ranked, start=1):
            scores[section_id] += 1.0 / (k + rank)
            if section_id not in docs:
                docs[section_id] = doc

    fused = [
        (sid, docs[sid], score)
        for sid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return fused

def hybrid_search(
    question: str,
    depth: int = RETRIEVAL_DEPTH,
) -> list[tuple[Document, float]]:
    bm25_results = indexer.search_bm25(question, k=depth)
    bm25_ranked: list[tuple[str, Document]] = [
        (doc.metadata["section_id"], doc) for doc, _score in bm25_results
    ]

    faiss_results = indexer.search_faiss(question, k=depth)
    faiss_ranked: list[tuple[str, Document]] = []
    seen_faiss_sections: set[str] = set()
    for chunk_doc, _distance in faiss_results:
        parent = indexer.resolve_parent_section(chunk_doc)
        sid = parent.metadata.get("section_id", chunk_doc.metadata.get("section_id", "unknown"))
        if sid not in seen_faiss_sections:
            seen_faiss_sections.add(sid)
            faiss_ranked.append((sid, parent))

    fused = _reciprocal_rank_fusion([bm25_ranked, faiss_ranked])
    
    log_retrieval_event(
        mode="hybrid",
        query=question,
        top_hits=[(sid, round(score, 6)) for sid, _, score in fused],
    )

    return [(doc, score) for _sid, doc, score in fused]

def query(question: str) -> dict:
    if indexer.vectorstore is None and indexer.bm25_index is None:
        return {
            "answer": "The knowledge base has not been indexed yet. Call POST /index first.",
            "sources": [],
        }

    ranked_sections = hybrid_search(question)

    if not ranked_sections:
        return {
            "answer": "I cannot confirm this from the knowledge base.",
            "sources": [],
        }

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

    top_k = ranked_sections[:3]

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
