import logging
from langchain.schema import Document

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a knowledge base Q&A assistant.  Answer ONLY using the CONTEXT provided below.

Rules:
1. Base every claim exclusively on the provided CONTEXT sections.  Do NOT use
   outside knowledge, guessing, or creative interpretation.
2. Cite your sources using the exact [Source: ...] tags shown before each
   context block.  Place citations inline, immediately after the fact they
   support.
3. If the CONTEXT does not contain enough information to answer the question,
   respond with: "I cannot confirm this from the knowledge base."
4. Never fabricate source IDs.  Only cite IDs that appear in the CONTEXT.
5. Keep answers concise and well-structured.
"""

def build_prompt(query: str, ranked_sections: list[tuple[Document, float]]) -> str:
    """Build a grounded prompt with source-cited context blocks."""
    if not ranked_sections:
        return f"CONTEXT:\n(no context)\n\nQUESTION:\n{query}"

    context_blocks: list[str] = []
    for doc, score in ranked_sections:
        source_id = doc.metadata.get("section_id", "unknown")
        context_blocks.append(
            f"[Source: {source_id}]\n{doc.page_content}"
        )

    context_text = "\n\n---\n\n".join(context_blocks)
    return f"CONTEXT:\n{context_text}\n\nQUESTION:\n{query}"

def log_retrieval_event(
    mode: str,
    query: str,
    top_hits: list[tuple[str, float]],
) -> None:
    """Structured audit log for every retrieval event."""
    logger.info(
        "RETRIEVAL mode=%s query=%r top_hits=%s",
        mode,
        query[:80],
        top_hits[:3],
    )
