"""FastAPI application entry point.

Mounts static files for the browser UI and loads persisted indexes on startup.
"""

import logging

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .indexer import load_vector_index
from .routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Hybrid RAG Knowledge Base Q&A Bot")
app.include_router(router)

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
def load_persisted_index():
    """Attempt to load persisted FAISS + BM25 indexes on startup."""
    try:
        files, sections = load_vector_index()
        if files:
            logger.info("Loaded persisted index: %d files, %d sections", files, sections)
        else:
            logger.info("No persisted index found — POST /index to build one")
    except Exception as exc:
        logger.warning("Skipping persisted index load: %s", exc)
