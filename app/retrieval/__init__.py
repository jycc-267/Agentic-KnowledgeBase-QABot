from .bm25_search import query as bm25_query
from .vector_search import query as vector_query
from .hybrid_search import query as hybrid_query

# A dictionary to route modes to functions
STRATEGIES = {
    "bm25": bm25_query,
    "vector": vector_query,
    "hybrid": hybrid_query,
}
