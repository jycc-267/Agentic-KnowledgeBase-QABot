import os

from langchain.schema import HumanMessage, SystemMessage
# pyrefly: ignore [missing-import]
from langchain_openai import ChatOpenAI
# pyrefly: ignore [missing-import]
from langchain_google_genai import ChatGoogleGenerativeAI

from . import indexer


SYSTEM_PROMPT = """
# TODO: Write the system prompt for the knowledge base Q&A assistant.
#
# Design decision: Hallucination defense for retrieved chunks.
#
# Hints:
# 1. Only answer using the provided CONTEXT.
# 2. Cite only exact source IDs shown in [Source: ...].
#    Each source ID uses filename#heading format.
# 3. Define fallback behavior when the context lacks the answer.
# 4. Explicitly prohibit guessing or outside knowledge.
"""

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0,
        )
    return _llm


def build_prompt(query: str, ranked_chunks: list) -> str:
    # TODO: Build the prompt from retrieved vector chunks.
    #
    # Design decision: Give the LLM enough context without flooding it.
    #
    # Hints:
    # 1. Include [Source: filename#heading] before each chunk.
    # 2. Include retrieval distance or score only for debugging.
    # 3. Use top-k chunks passed into this function.
    # 4. Place CONTEXT before QUESTION.
    return f"CONTEXT:\n(no context)\n\nQUESTION:\n{query}"


def query(question: str) -> dict:
    if indexer.vectorstore is None:
        return {
            "answer": "The knowledge base has not been indexed yet. Call POST /index first.",
            "sources": [],
        }

    ranked_chunks = indexer.search(question, k=3)
    if not ranked_chunks:
        return {
            "answer": "I cannot confirm from the knowledge base.",
            "sources": [],
        }

    response = get_llm().invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_prompt(question, ranked_chunks)),
    ])

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "heading": doc.metadata.get("heading", "unknown"),
            "score": round(float(score), 3),
            "content": doc.page_content[:240],
        }
        for doc, score in ranked_chunks
    ]

    return {
        "answer": response.content,
        "sources": sources,
    }
