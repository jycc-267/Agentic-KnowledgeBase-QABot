"""API routes and frontend serving.

Endpoints:
  GET  /          → Serve the browser UI (index.html)
  GET  /health    → Health check
  POST /index     → Rebuild dual indexes (BM25 + FAISS)
  POST /chat      → Hybrid retrieval + LLM answer
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .indexer import build_index
from .retrieval import STRATEGIES
from .schemas import ChatRequest, ChatResponse, IndexResponse
from evals.run_eval import run_evaluation, DEFAULT_EVAL_SET

logger = logging.getLogger(__name__)

router = APIRouter()
STATIC_DIR = Path(__file__).resolve().parent / "static"


@router.get("/")
def serve_frontend():
    """Serve the single-page browser UI."""
    index_html = STATIC_DIR / "index.html"
    if not index_html.exists():
        raise HTTPException(status_code=404, detail="Frontend not built yet")
    return FileResponse(str(index_html), media_type="text/html")


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/index", response_model=IndexResponse)
def index_docs():
    try:
        files_count, sections_count = build_index()
    except Exception as exc:
        logger.error("Index build failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return IndexResponse(files_indexed=files_count, sections_indexed=sections_count)


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if req.mode not in STRATEGIES:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {req.mode}")
    query_fn = STRATEGIES[req.mode]
    return query_fn(req.query)

@router.get("/evaluate")
def evaluate(k: int = 3):
    try:
        summary = run_evaluation(DEFAULT_EVAL_SET, k)
        return summary
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
