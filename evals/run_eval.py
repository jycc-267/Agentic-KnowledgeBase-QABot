"""Evaluation runner for the Hybrid RAG retrieval system.

Measures:
- Recall@K: Fraction of expected sources that appear in the top-K results.
- MRR (Mean Reciprocal Rank): Average of 1/rank for the first relevant hit.
- Source Hit Rate: Fraction of queries where at least one expected source appears.

Usage:
    uv run python -m evals.run_eval
    uv run python -m evals.run_eval --eval-set evals/eval_set.json --k 3
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from app.indexer import build_index, load_vector_index
from app.retrieval import bm25_search, vector_search, hybrid_search


STRATEGIES = {
    "BM25": bm25_search.search,
    "Vector": vector_search.search,
    "Hybrid": hybrid_search.search,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_EVAL_SET = PROJECT_ROOT / "evals" / "eval_set.json"


def load_eval_set(path: Path) -> list[dict]:
    """Load the evaluation queries from JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_query(
    search_fn,
    query_text: str,
    expected_sources: list[str],
    k: int,
) -> dict:
    """Evaluate a single query against the retrieval system.

    Returns a dict with recall_at_k, reciprocal_rank, hit, and retrieved_sources.
    """
    results = search_fn(query_text, depth=k)
    
    retrieved_details = [
        {
            "section_id": doc.metadata.get("section_id", "unknown"),
            "heading": doc.metadata.get("heading", "unknown"),
            "score": round(float(score), 4)
        }
        for doc, score in results[:k]
    ]
    retrieved_ids = [d["section_id"] for d in retrieved_details]

    # For FM4 (out-of-scope): expected_sources is empty → success if no results
    if not expected_sources:
        is_correct_fallback = len(results) == 0 or results[0][1] < 0.005
        return {
            "retrieved_sources": retrieved_ids,
            "retrieved_details": retrieved_details,
            "recall_at_k": 1.0 if is_correct_fallback else 0.0,
            "reciprocal_rank": 1.0 if is_correct_fallback else 0.0,
            "hit": is_correct_fallback,
            "is_fallback_test": True,
        }

    # Recall@K: what fraction of expected sources were retrieved?
    hits = sum(1 for src in expected_sources if src in retrieved_ids)
    recall = hits / len(expected_sources) if expected_sources else 0.0

    # Reciprocal Rank: 1 / (rank of first relevant result)
    rr = 0.0
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in expected_sources:
            rr = 1.0 / rank
            break

    return {
        "retrieved_sources": retrieved_ids,
        "retrieved_details": retrieved_details,
        "recall_at_k": recall,
        "reciprocal_rank": rr,
        "hit": hits > 0,
        "is_fallback_test": False,
    }


def run_evaluation(eval_set_path: Path, k: int) -> dict:
    """Run the full evaluation suite and compute aggregate metrics."""
    eval_cases = load_eval_set(eval_set_path)

    # Ensure indexes are built
    try:
        files, sections = load_vector_index()
        if not files:
            logger.info("No persisted index — building from scratch...")
            files, sections = build_index()
    except Exception:
        logger.info("Building index from scratch...")
        files, sections = build_index()

    logger.info("Index ready: %d files, %d sections", files, sections)

    summary = {"k": k, "num_queries": len(eval_cases), "strategies": {}}

    for strategy_name, search_fn in STRATEGIES.items():
        print("\n" + "=" * 80)
        print(f"  EVALUATING STRATEGY: {strategy_name}  (K={k}, {len(eval_cases)} queries)")
        print("=" * 80)
        
        results: list[dict] = []
        total_recall = 0.0
        total_rr = 0.0
        total_hits = 0
        n = len(eval_cases)

        for i, case in enumerate(eval_cases, start=1):
            query_text = case["query"]
            expected = case.get("expected_sources", [])
            fm = case.get("failure_mode", "—")

            result = evaluate_query(search_fn, query_text, expected, k=k)
            results.append({**case, **result})

            total_recall += result["recall_at_k"]
            total_rr += result["reciprocal_rank"]
            total_hits += 1 if result["hit"] else 0
            
            # Print failures only to keep output clean, or print all
            if not result["hit"]:
                print(f"\n❌ [{i}/{n}] {query_text} (FM: {fm})")
                print(f"   Expected:  {expected}")
                print(f"   Retrieved: {result['retrieved_sources']}")

        avg_recall = total_recall / n if n else 0.0
        mrr = total_rr / n if n else 0.0
        source_hit_rate = total_hits / n if n else 0.0
        
        summary["strategies"][strategy_name] = {
            "recall_at_k": round(avg_recall, 4),
            "mrr": round(mrr, 4),
            "source_hit_rate": round(source_hit_rate, 4),
            "details": results,
        }

    print("\n" + "=" * 80)
    print("  COMPARATIVE AGGREGATE METRICS")
    print("=" * 80)
    print(f"{'Strategy':<10} | {'Recall@' + str(k):<10} | {'MRR':<10} | {'Hit Rate':<10}")
    print("-" * 50)
    for name, stats in summary["strategies"].items():
        print(f"{name:<10} | {stats['recall_at_k']:<10.4f} | {stats['mrr']:<10.4f} | {stats['source_hit_rate']:<10.4f}")
    print("=" * 80 + "\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run hybrid RAG evaluation")
    parser.add_argument(
        "--eval-set", type=Path, default=DEFAULT_EVAL_SET,
        help="Path to eval_set.json",
    )
    parser.add_argument(
        "--k", type=int, default=3,
        help="Top-K for Recall@K metric (default: 3)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Optional path to write full results JSON",
    )
    args = parser.parse_args()

    summary = run_evaluation(args.eval_set, args.k)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("Results written to %s", args.output)


if __name__ == "__main__":
    main()
