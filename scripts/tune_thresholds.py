"""Systematic threshold & parameter tuner for the Hybrid RAG retrieval system.

Approach:
1. Collect raw (unfiltered) scores from each retrieval strategy across the eval set.
2. Separate scores into "in-scope" (expected_sources is non-empty) vs "out-of-scope" (FM4).
3. Identify the score gap between the two populations and compute an optimal threshold.
4. For hybrid search, additionally sweep RRF_K values to find the best fusion parameter.
5. Print a full analysis report with recommended values and safety margins.

Usage:
    uv run python scripts/tune_thresholds.py
    uv run python scripts/tune_thresholds.py --eval-set scripts/eval_set.json --k 3
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document

from app.indexer import build_index, load_vector_index, search_bm25, search_faiss
from app.indexer import resolve_parent_section

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_EVAL_SET = PROJECT_ROOT / "scripts" / "eval_set.json"


# ---------------------------------------------------------------------------
# Raw score collection (no threshold filtering)
# ---------------------------------------------------------------------------

def _collect_bm25_scores(
    query_text: str,
    depth: int,
) -> list[tuple[str, float]]:
    """Return raw BM25 (section_id, score) pairs with no filtering."""
    results = search_bm25(query_text, k=depth)
    return [
        (doc.metadata.get("section_id", "unknown"), float(score))
        for doc, score in results
    ]


def _collect_vector_scores(
    query_text: str,
    depth: int,
) -> list[tuple[str, float]]:
    """Return raw vector (section_id, similarity) pairs with no filtering.

    Converts FAISS L2 distance to similarity: 1/(1+distance).
    Deduplicates by parent section (keeps the best score per section).
    """
    faiss_results = search_faiss(query_text, k=depth)
    seen: set[str] = set()
    out: list[tuple[str, float]] = []
    for chunk_doc, distance in faiss_results:
        parent = resolve_parent_section(chunk_doc)
        sid = parent.metadata.get(
            "section_id",
            chunk_doc.metadata.get("section_id", "unknown"),
        )
        if sid not in seen:
            seen.add(sid)
            similarity = 1.0 / (1.0 + float(distance))
            out.append((sid, similarity))
    return out


def _collect_hybrid_raw(
    query_text: str,
    depth: int,
) -> dict:
    """Return both raw BM25 and raw FAISS lists for hybrid analysis."""
    bm25 = _collect_bm25_scores(query_text, depth)
    vector = _collect_vector_scores(query_text, depth)
    return {"bm25": bm25, "vector": vector}


# ---------------------------------------------------------------------------
# RRF fusion with configurable k (for sweep)
# ---------------------------------------------------------------------------

def _rrf_fuse(
    bm25_ranked: list[tuple[str, Document]],
    faiss_ranked: list[tuple[str, Document]],
    rrf_k: int,
) -> list[tuple[str, float]]:
    """Fuse two ranked lists with RRF using the given k constant."""
    scores: dict[str, float] = defaultdict(float)
    for rank, (sid, _doc) in enumerate(bm25_ranked, start=1):
        scores[sid] += 1.0 / (rrf_k + rank)
    for rank, (sid, _doc) in enumerate(faiss_ranked, start=1):
        scores[sid] += 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Threshold analysis helpers
# ---------------------------------------------------------------------------

def _find_optimal_threshold(
    in_scope_scores: list[float],
    oos_scores: list[float],
    candidates: list[float],
) -> dict:
    """Find the threshold that maximizes separation between in-scope and OOS.

    For each candidate threshold T:
        - true_positives  = in-scope scores >= T
        - true_negatives  = out-of-scope scores < T
        - false_negatives = in-scope scores < T   (we lose a good hit)
        - false_positives = out-of-scope scores >= T  (we let junk through)

    We pick the T that maximizes (TP_rate + TN_rate) / 2 (balanced accuracy).
    """
    best: dict = {"threshold": 0.0, "balanced_accuracy": 0.0}

    for t in candidates:
        tp = sum(1 for s in in_scope_scores if s >= t)
        fn = sum(1 for s in in_scope_scores if s < t)
        tn = sum(1 for s in oos_scores if s < t)
        fp = sum(1 for s in oos_scores if s >= t)

        tp_rate = tp / (tp + fn) if (tp + fn) else 1.0
        tn_rate = tn / (tn + fp) if (tn + fp) else 1.0
        balanced = (tp_rate + tn_rate) / 2.0

        if balanced > best["balanced_accuracy"]:
            best = {
                "threshold": t,
                "balanced_accuracy": round(balanced, 4),
                "tp_rate": round(tp_rate, 4),
                "tn_rate": round(tn_rate, 4),
                "tp": tp,
                "fn": fn,
                "tn": tn,
                "fp": fp,
            }

    return best


def _eval_recall_at_threshold(
    cases: list[dict],
    all_raw_scores: list[list[tuple[str, float]]],
    threshold: float,
    k: int,
) -> dict:
    """Compute Recall@K, MRR, and Hit Rate given a threshold filter."""
    total_recall = 0.0
    total_rr = 0.0
    total_hits = 0
    n = len(cases)

    for case, raw_scores in zip(cases, all_raw_scores):
        expected = case.get("expected_sources", [])
        filtered = [(sid, s) for sid, s in raw_scores if s >= threshold][:k]
        retrieved_ids = [sid for sid, _s in filtered]

        if not expected:
            is_correct = len(filtered) == 0
            total_recall += 1.0 if is_correct else 0.0
            total_rr += 1.0 if is_correct else 0.0
            total_hits += 1 if is_correct else 0
            continue

        hits = sum(1 for src in expected if src in retrieved_ids)
        recall = hits / len(expected)
        rr = 0.0
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in expected:
                rr = 1.0 / rank
                break

        total_recall += recall
        total_rr += rr
        total_hits += 1 if hits > 0 else 0

    return {
        "recall_at_k": round(total_recall / n, 4) if n else 0.0,
        "mrr": round(total_rr / n, 4) if n else 0.0,
        "hit_rate": round(total_hits / n, 4) if n else 0.0,
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_tuning(eval_set_path: Path, k: int) -> None:
    """Run the full threshold tuning analysis."""
    eval_cases = json.loads(eval_set_path.read_text(encoding="utf-8"))

    # Ensure indexes are loaded
    try:
        files, sections = load_vector_index()
        if not files:
            files, sections = build_index()
    except Exception:
        files, sections = build_index()

    print(f"\n📊 Index ready: {files} files, {sections} sections")
    print(f"📋 Eval set: {len(eval_cases)} queries (K={k})\n")

    # ------------------------------------------------------------------
    # Phase 1: Collect raw scores
    # ------------------------------------------------------------------
    bm25_raw: list[list[tuple[str, float]]] = []
    vector_raw: list[list[tuple[str, float]]] = []

    for case in eval_cases:
        q = case["query"]
        bm25_raw.append(_collect_bm25_scores(q, depth=k))
        vector_raw.append(_collect_vector_scores(q, depth=k))

    # ------------------------------------------------------------------
    # Phase 2: Separate in-scope vs out-of-scope (FM4) top-1 scores
    # ------------------------------------------------------------------
    for strategy_name, raw_data in [("BM25", bm25_raw), ("Vector", vector_raw)]:
        print("=" * 80)
        print(f"  STRATEGY: {strategy_name}")
        print("=" * 80)

        in_scope_top1: list[float] = []
        oos_top1: list[float] = []
        in_scope_all: list[float] = []
        oos_all: list[float] = []

        for case, scores in zip(eval_cases, raw_data):
            expected = case.get("expected_sources", [])
            all_s = [s for _, s in scores]

            if not expected:  # FM4 out-of-scope
                oos_all.extend(all_s)
                if all_s:
                    oos_top1.append(max(all_s))
                else:
                    oos_top1.append(0.0)
            else:
                in_scope_all.extend(all_s)
                if all_s:
                    in_scope_top1.append(max(all_s))
                else:
                    in_scope_top1.append(0.0)

        # Print score distributions
        print(f"\n  {'Population':<20} | {'Count':<6} | {'Min':<8} | {'Max':<8} | {'Mean':<8} | {'Median':<8}")
        print("  " + "-" * 68)
        for label, data in [
            ("In-scope (top-1)", in_scope_top1),
            ("In-scope (all)", in_scope_all),
            ("Out-of-scope (top-1)", oos_top1),
            ("Out-of-scope (all)", oos_all),
        ]:
            if data:
                s = sorted(data)
                mean = sum(s) / len(s)
                median = s[len(s) // 2]
                print(f"  {label:<20} | {len(s):<6} | {min(s):<8.4f} | {max(s):<8.4f} | {mean:<8.4f} | {median:<8.4f}")
            else:
                print(f"  {label:<20} | 0      | —        | —        | —        | —")

        # Print per-query detail
        print(f"\n  Per-query score breakdown:")
        print(f"  {'Query':<55} | {'Type':<5} | {'Top Score':<10} | {'Top Section':<35}")
        print("  " + "-" * 112)
        for case, scores in zip(eval_cases, raw_data):
            expected = case.get("expected_sources", [])
            fm = case.get("failure_mode", "—")
            q_short = case["query"][:53]
            qtype = "OOS" if not expected else "IS"
            if scores:
                top_sid, top_score = scores[0]
                print(f"  {q_short:<55} | {qtype:<5} | {top_score:<10.4f} | {top_sid:<35}")
            else:
                print(f"  {q_short:<55} | {qtype:<5} | {'(none)':<10} | {'—':<35}")

        # Find optimal threshold via sweep
        if strategy_name == "BM25":
            candidates = [round(x * 0.1, 2) for x in range(0, 80)]  # 0.0 to 8.0
        else:
            candidates = [round(x * 0.01, 3) for x in range(40, 80)]  # 0.40 to 0.80

        optimal = _find_optimal_threshold(in_scope_top1, oos_top1, candidates)

        # Compute the actual gap
        in_scope_min = min(in_scope_top1) if in_scope_top1 else 0.0
        oos_max = max(oos_top1) if oos_top1 else 0.0
        gap = in_scope_min - oos_max

        print(f"\n  📐 Score Gap Analysis:")
        print(f"     Lowest in-scope top-1:  {in_scope_min:.4f}")
        print(f"     Highest OOS top-1:      {oos_max:.4f}")
        print(f"     Gap:                    {gap:.4f} {'✅ clean separation' if gap > 0 else '⚠️  overlapping!'}")

        print(f"\n  🎯 Optimal Threshold: {optimal['threshold']:.4f}")
        print(f"     Balanced Accuracy: {optimal['balanced_accuracy']:.4f}")
        print(f"     TP rate (in-scope kept):  {optimal.get('tp_rate', 0):.4f} ({optimal.get('tp', 0)}/{optimal.get('tp', 0) + optimal.get('fn', 0)})")
        print(f"     TN rate (OOS rejected):   {optimal.get('tn_rate', 0):.4f} ({optimal.get('tn', 0)}/{optimal.get('tn', 0) + optimal.get('fp', 0)})")

        # Recommended threshold with safety margin (lean toward filtering OOS)
        if gap > 0:
            # Place threshold at midpoint of gap
            recommended = round((in_scope_min + oos_max) / 2.0, 4)
        else:
            recommended = optimal["threshold"]

        print(f"     ➡️  Recommended (midpoint of gap): {recommended:.4f}")

        # Sweep a few candidates and show metrics
        sweep_thresholds = sorted({
            0.0,
            recommended,
            optimal["threshold"],
            round(oos_max, 4),
            round(in_scope_min, 4),
            round(oos_max + 0.01, 4),
            round(in_scope_min - 0.01, 4),
        })

        print(f"\n  📊 Recall@{k} / MRR / Hit Rate at various thresholds:")
        print(f"  {'Threshold':<12} | {'Recall@' + str(k):<10} | {'MRR':<10} | {'Hit Rate':<10} | Note")
        print("  " + "-" * 70)
        for t in sweep_thresholds:
            metrics = _eval_recall_at_threshold(eval_cases, raw_data, t, k)
            note = ""
            if t == recommended:
                note = "← RECOMMENDED"
            elif t == optimal["threshold"]:
                note = "← optimal balanced acc"
            print(f"  {t:<12.4f} | {metrics['recall_at_k']:<10.4f} | {metrics['mrr']:<10.4f} | {metrics['hit_rate']:<10.4f} | {note}")

        print()

    # ------------------------------------------------------------------
    # Phase 3: Hybrid RRF_K sweep
    # ------------------------------------------------------------------
    print("=" * 80)
    print("  HYBRID: RRF_K SWEEP")
    print("=" * 80)

    rrf_k_candidates = [10, 20, 30, 40, 50, 60, 80, 100]
    faiss_threshold_candidates = [round(x * 0.01, 3) for x in range(40, 80)]

    # Pre-collect FAISS parent-resolved ranked lists (with scores for threshold filtering)
    faiss_parent_resolved: list[list[tuple[str, Document, float]]] = []
    bm25_doc_lists: list[list[tuple[str, Document]]] = []

    for case in eval_cases:
        q = case["query"]
        # BM25 ranked list (section_id, Document)
        bm25_results = search_bm25(q, k=k)
        bm25_doc_lists.append([
            (doc.metadata["section_id"], doc) for doc, _s in bm25_results
        ])

        # FAISS with parent resolution + similarity
        faiss_results = search_faiss(q, k=k)
        seen: set[str] = set()
        resolved: list[tuple[str, Document, float]] = []
        for chunk_doc, distance in faiss_results:
            parent = resolve_parent_section(chunk_doc)
            sid = parent.metadata.get(
                "section_id",
                chunk_doc.metadata.get("section_id", "unknown"),
            )
            if sid not in seen:
                seen.add(sid)
                similarity = 1.0 / (1.0 + float(distance))
                resolved.append((sid, parent, similarity))
        faiss_parent_resolved.append(resolved)

    print(f"\n  Sweeping {len(rrf_k_candidates)} RRF_K values × {len(faiss_threshold_candidates)} FAISS thresholds...\n")

    best_hybrid: dict = {"score": 0.0}

    # Collect per-config results for a summary table
    rrf_results: list[dict] = []

    for rrf_k in rrf_k_candidates:
        for ft in faiss_threshold_candidates:
            # For each eval query, fuse BM25 + threshold-filtered FAISS via RRF
            hybrid_scores: list[list[tuple[str, float]]] = []

            for i, case in enumerate(eval_cases):
                bm25_ranked = bm25_doc_lists[i]
                faiss_ranked = [
                    (sid, doc)
                    for sid, doc, sim in faiss_parent_resolved[i]
                    if sim >= ft
                ]
                fused = _rrf_fuse(bm25_ranked, faiss_ranked, rrf_k)
                hybrid_scores.append(fused)

            # Evaluate at a range of RRF score thresholds
            # RRF scores are typically small: 1/(k+rank), so sweep accordingly
            rrf_score_candidates = [round(x * 0.001, 4) for x in range(0, 50)]

            for rrf_thresh in rrf_score_candidates:
                metrics = _eval_recall_at_threshold(
                    eval_cases, hybrid_scores, rrf_thresh, k,
                )
                composite = (
                    metrics["recall_at_k"] * 0.4
                    + metrics["mrr"] * 0.3
                    + metrics["hit_rate"] * 0.3
                )

                entry = {
                    "rrf_k": rrf_k,
                    "faiss_threshold": ft,
                    "confidence_threshold": rrf_thresh,
                    **metrics,
                    "composite": round(composite, 4),
                }
                rrf_results.append(entry)

                if composite > best_hybrid.get("score", 0.0):
                    best_hybrid = {"score": composite, **entry}

    # Print top 10 hybrid configs
    rrf_results.sort(key=lambda x: x["composite"], reverse=True)
    top_n = 10
    print(f"  Top {top_n} Hybrid configurations (by composite = 0.4×Recall + 0.3×MRR + 0.3×HitRate):\n")
    print(f"  {'RRF_K':<8} | {'FAISS_T':<9} | {'Conf_T':<9} | {'Recall':<8} | {'MRR':<8} | {'HitRate':<8} | {'Composite':<10}")
    print("  " + "-" * 78)
    for entry in rrf_results[:top_n]:
        print(
            f"  {entry['rrf_k']:<8} | "
            f"{entry['faiss_threshold']:<9.3f} | "
            f"{entry['confidence_threshold']:<9.4f} | "
            f"{entry['recall_at_k']:<8.4f} | "
            f"{entry['mrr']:<8.4f} | "
            f"{entry['hit_rate']:<8.4f} | "
            f"{entry['composite']:<10.4f}"
        )

    # ------------------------------------------------------------------
    # Phase 4: Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  RECOMMENDED CONFIGURATION")
    print("=" * 80)
    print(f"""
  These values were derived by analyzing the score gap between in-scope and
  out-of-scope queries, then selecting thresholds at the midpoint of the gap
  to maximize safety margin.

  Hybrid parameters were found by grid-sweeping RRF_K × FAISS_THRESHOLD ×
  CONFIDENCE_THRESHOLD and ranking by composite score.

  ┌──────────────────────────────────────────────────────────────────────┐
  │  bm25_search.py                                                    │
  │    CONFIDENCE_THRESHOLD  = (see BM25 analysis above)               │
  │                                                                    │
  │  vector_search.py                                                  │
  │    CONFIDENCE_THRESHOLD  = (see Vector analysis above)             │
  │                                                                    │
  │  hybrid_search.py                                                  │
  │    RRF_K                 = {best_hybrid.get('rrf_k', '?'):<10}                              │
  │    FAISS_THRESHOLD       = {best_hybrid.get('faiss_threshold', '?'):<10}                              │
  │    CONFIDENCE_THRESHOLD  = {best_hybrid.get('confidence_threshold', '?'):<10}                              │
  └──────────────────────────────────────────────────────────────────────┘
""")

    print("  ⚠️  Caution: These values are tuned to the current eval set.")
    print("     Add more diverse queries to avoid overfitting to a small sample.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Tune retrieval thresholds and parameters",
    )
    parser.add_argument(
        "--eval-set", type=Path, default=DEFAULT_EVAL_SET,
        help="Path to eval_set.json",
    )
    parser.add_argument(
        "--k", type=int, default=3,
        help="Top-K for evaluation (default: 3)",
    )
    args = parser.parse_args()
    run_tuning(args.eval_set, args.k)


if __name__ == "__main__":
    main()
