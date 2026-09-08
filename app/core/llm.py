import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-2"
_embeddings = None
_llm = None


def get_embeddings():
    """Lazy-initialise Google Gemini embeddings client."""
    global _embeddings
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is not set in the server environment")
    if _embeddings is None:
        _embeddings = GoogleGenerativeAIEmbeddings(
            model=GEMINI_EMBEDDING_MODEL,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            request_options={
                "timeout": 20,
                "max_retries": 1,
            },
        )
    return _embeddings


def get_llm():
    """Lazy-initialise the Gemini chat model."""
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,
        )
    return _llm
