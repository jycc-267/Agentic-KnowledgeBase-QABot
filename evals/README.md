# Threshold Tuning Methodology & Results

## Problem

The retrieval system has several tunable parameters across 3 strategies, and the previous values were either hand-picked or arbitrary:

| Parameter | Module | Old Value | How it was chosen |
|-----------|--------|-----------|-------------------|
| `CONFIDENCE_THRESHOLD` | `vector_search.py` | `0.56` | Eyeballed from eval output |
| `CONFIDENCE_THRESHOLD` | `bm25_search.py` | `5.0` | Arbitrary round number |
| `RRF_K` | `hybrid_search.py` | `60` | Textbook default from the RRF paper |
| `FAISS_THRESHOLD` | `hybrid_search.py` | `0.56` | Copy-pasted from vector search |
| `CONFIDENCE_THRESHOLD` | `hybrid_search.py` | `0.005` | Arbitrary small number |

We need a systematic, repeatable way to derive these values from data.

## Methodology

### How the Tuner Collects Raw Scores

The tuner ([tune_thresholds.py](../../evals/tune_thresholds.py)) bypasses the retrieval modules (`bm25_search.py`, `vector_search.py`, `hybrid_search.py`) entirely. It calls the **indexer-level** functions directly:

- `indexer.search_bm25()` — returns raw `(Document, bm25_score)` pairs with only a `score > 0` filter
- `indexer.search_faiss()` — returns raw `(Document, L2_distance)` pairs with no filtering
- `indexer.resolve_parent_section()` — resolves chunks back to parent sections

This is different from calling `bm25_search.search()` or `vector_search.search()`, which are the **retrieval-layer** functions in each approach's module. Those retrieval-layer `search()` functions wrap the indexer calls and (after tuning) apply threshold filtering before returning results. By going directly to the indexer, the tuner sees the complete, unfiltered picture.

**Where thresholds are applied in the codebase:**

Each retrieval module has two functions:
- `search()` — the retrieval-layer function that fetches and filters results (this is what `run_eval.py` calls)
- `query()` — the full RAG pipeline that calls `search()`, applies a secondary confidence check, calls the LLM, and formats the answer (this is what the `/chat` API endpoint calls)

After tuning, we moved threshold filtering *into* each module's `search()` function. For example, `vector_search.search()` now filters out results below `CONFIDENCE_THRESHOLD=0.5648` before returning. This means both `query()` (the API path) and `run_eval.py` (the eval path) benefit from the same filtering.

---

### Phase 1: Raw Score Collection

For each of the 16 eval queries, we call `indexer.search_bm25()` and `indexer.search_faiss()` directly, recording every `(section_id, raw_score)` pair they return.

**Example output for "What is the meaning of life?" (an out-of-scope query):**

| Engine | Section ID | Raw Score |
|--------|------------|-----------|
| BM25 | *(no results — no keyword overlap)* | — |
| Vector | `account_help.md#account-help` | 0.5413 |
| Vector | `shipping_faq.md#shipping-faq` | 0.5257 |
| Vector | `refund_policy.md#refund-policy` | 0.5149 |

BM25 returns nothing because no tokens match. But the vector engine *always* returns something — FAISS finds the K nearest neighbors regardless of how irrelevant they are. The scores above are low, but not zero. We need a threshold to tell the system "these are too low to trust."

### Phase 2: Score Distribution Analysis

**The core idea:** If we can show that scores from "good" queries (where the KB has an answer) are consistently higher than scores from "bad" queries (where the KB has no answer), then there exists a score boundary we can use as a threshold.

We split all eval queries into two groups:

- **In-scope (IS):** 14 queries that *should* find a match (e.g., "How long does a refund take?" → expects `refund_policy.md#refund-timeline`)
- **Out-of-scope (OOS):** 2 queries that should return *nothing* (e.g., "What is the meaning of life?" → expects `[]`)

For each query, we look at the **top-1 score** (the highest-scoring result). Then we compare the distributions:

**Vector Search Example:**

```
In-scope top-1 scores:
  "How long does standard delivery take?"         → 0.7437
  "Is expedited shipping refundable...?"           → 0.7621
  "Is it 24 hours?"                                → 0.5725 (lowest)
  ... (14 queries, range: 0.5725 — 0.7621)

Out-of-scope top-1 scores:
  "What is the meaning of life?"                   → 0.5413
  "Do you offer cryptocurrency payments?"          → 0.5571 (highest OOS)
  ... (2 queries, range: 0.5413 — 0.5571)
```

Visually, on a number line:

```
0.50    0.52    0.54    0.56    0.58    0.60    0.62    0.64 ...
                  |       |       |
                 OOS    GAP     IS
              (≤0.5571)  ↑   (≥0.5725)
                     threshold
                     goes here
```

The **gap** is the distance between the lowest in-scope score (0.5725) and the highest OOS score (0.5571):

```
Gap = 0.5725 − 0.5571 = 0.0154
```

A positive gap means the two populations don't overlap — there's a clean boundary. We place the threshold at the **midpoint** of the gap to maximize the safety margin on both sides:

```
Threshold = (0.5725 + 0.5571) / 2 = 0.5648
```

This gives us 0.0077 of margin before we'd accidentally filter out a real result, and 0.0077 of margin before we'd let through an OOS result.

**What if the gap is negative (overlapping distributions)?**

A negative gap means some OOS queries score higher than some in-scope queries — there is no clean separation. In this case, we **fall back to the optimal balanced accuracy threshold** from Phase 3. This is the threshold that makes the best trade-off: it accepts that some errors are unavoidable, and picks the point that minimizes the *total* errors (both letting junk through and throwing away good results). In a severe overlap, this signals that the retrieval engine alone can't distinguish in-scope from OOS, and you'd need to add a second-stage filter (e.g., an LLM relevance check) or improve the embeddings.

### Phase 3: Optimal Threshold Selection

Phase 2 gives us a good starting point (the midpoint), but we validate it rigorously by sweeping many candidate thresholds and measuring what happens at each one.

**How candidates are chosen** depends on the score scale of each engine:

- **BM25** scores range from ~0 to ~6, so we sweep `0.0, 0.1, 0.2, ..., 7.9` (80 candidates at step 0.1)
- **Vector** similarity scores range from ~0.5 to ~0.8, so we sweep `0.40, 0.41, 0.42, ..., 0.79` (40 candidates at step 0.01)

These ranges are intentionally wide to cover any plausible threshold, not just the gap region.

**For each candidate threshold, we classify every query result as one of four outcomes:**

| | Query *should* match (IS) | Query should *not* match (OOS) |
|---|---|---|
| **Score ≥ threshold** (we keep it) | ✅ **True Positive (TP)** — Correctly kept a real match | ❌ **False Positive (FP)** — Incorrectly let junk through |
| **Score < threshold** (we filter it) | ❌ **False Negative (FN)** — Accidentally threw away a real match | ✅ **True Negative (TN)** — Correctly rejected noise |

**In plain English:**

- **TP Rate** (True Positive Rate) = "Of all the queries that *should* find a result, what fraction actually survive the threshold?" If TP Rate = 1.0, we never accidentally throw away a good result.
- **TN Rate** (True Negative Rate) = "Of all the queries that *should not* find a result, what fraction get correctly filtered out?" If TN Rate = 1.0, we never let noise through.
- **Balanced Accuracy** = average of TP Rate and TN Rate. It's the single number that tells us "how well does this threshold simultaneously (a) keep good results and (b) reject bad results?" We use the *balanced* version because we have many more IS queries (14) than OOS queries (2), and we don't want the IS queries to dominate.

**Example sweep for Vector search:**

| Threshold | TP Rate | TN Rate | Balanced Acc | Effect |
|-----------|---------|---------|--------------|--------|
| 0.50 | 14/14 = 1.00 | 0/2 = 0.00 | 0.50 | Keeps everything — OOS leaks through |
| 0.55 | 14/14 = 1.00 | 0/2 = 0.00 | 0.50 | Still too low |
| 0.56 | 14/14 = 1.00 | 2/2 = 1.00 | **1.00** | ✅ Perfect — first value above all OOS scores |
| 0.5648 | 14/14 = 1.00 | 2/2 = 1.00 | **1.00** | ✅ Our midpoint pick |
| 0.58 | 13/14 = 0.93 | 2/2 = 1.00 | 0.96 | Too aggressive — loses "Is it 24 hours?" (0.5725) |

**Cross-validation against retrieval metrics:** After finding the optimal threshold, we validate it against actual Recall@K, MRR, and Hit Rate. This happens in the tuner where `_eval_recall_at_threshold()` is called with a set of key thresholds (0.0, the recommended midpoint, the optimal balanced-accuracy threshold, the OOS max, the IS min, etc.). This confirms the threshold doesn't accidentally degrade retrieval quality on the full eval set — for example, we verify that the recommended threshold still achieves Recall@3 = 1.0.

### Phase 4: Hybrid Grid Sweep

Phases 2–3 tune a **single threshold** for BM25 and Vector search independently. But hybrid search is more complex because it has **three interacting parameters**:

1. **`RRF_K`** — The smoothing constant in the RRF formula: `score += 1/(RRF_K + rank)`. A smaller `RRF_K` amplifies the difference between rank 1 and rank 2. A larger `RRF_K` flattens the differences.
2. **`FAISS_THRESHOLD`** — Applied *before* fusion: "don't let FAISS contribute a result to the RRF merge unless its similarity score is at least this high." This prevents noisy vector matches from polluting the fused ranking.
3. **`CONFIDENCE_THRESHOLD`** — Applied *after* fusion: "if the final fused RRF score is below this, return nothing."

**Why FAISS_THRESHOLD in hybrid (0.68) differs from the pure vector threshold (0.5648):**

These thresholds serve fundamentally different roles:

- **Pure vector's threshold (0.5648)** is the *only* filter — it must keep everything that might be relevant, because vector search is the sole retrieval source. If it filters too aggressively, the user gets nothing.
- **Hybrid's FAISS_THRESHOLD (0.68)** is a *pre-fusion quality gate* — it controls which FAISS results are "good enough" to enter the RRF merge alongside BM25 results. If FAISS misses a result due to the higher threshold, BM25 can still independently surface it. The safety net of having two retrieval sources allows the FAISS gate to be stricter.

This is why Phases 2–3 (which tune single-engine thresholds) don't directly apply to hybrid — the hybrid parameters interact and need to be tuned together.

Because these three parameters interact (a stricter `FAISS_THRESHOLD` means fewer results enter RRF, which changes the RRF scores, which changes what `CONFIDENCE_THRESHOLD` filters), we can't tune them independently. Instead, Phase 4 tries **every combination** in a grid:

```
8 RRF_K values × 40 FAISS thresholds × 50 confidence thresholds = 16,000 configurations
```

For each of the 16,000 configurations, we simulate: "if hybrid search used these exact parameter values, what would Recall@3, MRR, and Hit Rate be across all 16 eval queries?" We rank configurations by a **composite score**:

```
Composite = 0.4 × Recall@K + 0.3 × MRR + 0.3 × Hit Rate
```

**What this composite score expresses:** It balances three complementary retrieval quality goals. The metrics are defined as follows:

- **Recall@K (weight 0.4)** — "Did we find *all* the expected sources?" Gets the highest weight because missing a relevant source is the worst failure mode in a knowledge base QA system — the user gets an incomplete or wrong answer.
  - *Formula:* `(Number of expected sources found in top K) / (Total number of expected sources for the query)`
- **MRR (weight 0.3)** — "Was the *first* relevant result ranked highly?" Matters because the LLM prompt puts top-ranked context first, and the top result disproportionately influences the generated answer.
  - *Formula:* `1 / (Rank of the first relevant result)`. If no relevant result is found, MRR is 0. Averaged across all queries.
- **Hit Rate (weight 0.3)** — "Did we find *at least one* relevant source?" A baseline safety check — the minimum bar for a useful retrieval.
  - *Formula:* `1 if (at least one expected source is in top K) else 0`. Averaged across all queries.

The weights (0.4/0.3/0.3) reflect that Recall is slightly more important than ranking quality, but all three matter. These weights are a judgment call — if ranking quality were paramount (e.g., showing a single-result card UI), you'd weight MRR higher.

---

## Results

### BM25 Score Gap

```
In-scope top-1 scores:  min=1.5088, max=5.8615
Out-of-scope top-1:     0.0000 (no matches at all)

Gap = 1.5088 ✅ very clean separation
Tuner recommended threshold: 0.7544 (midpoint of gap)
```

> [!NOTE]
> BM25 naturally handles OOS queries because exact keyword matching returns zero results when no tokens overlap. The `> 0` filter in `indexer.search_bm25()` already handles this, so the `CONFIDENCE_THRESHOLD` in `bm25_search.py` is mainly a secondary safeguard.

### Vector Search Score Gap

```
In-scope top-1 scores:  min=0.5725, max=0.7621, mean=0.6761
Out-of-scope top-1:     min=0.5413, max=0.5571, mean=0.5492

Gap = 0.5725 - 0.5571 = 0.0154 ✅ clean but narrow separation
Midpoint threshold = (0.5725 + 0.5571) / 2 = 0.5648
```

### Hybrid Grid Sweep Winner

```
RRF_K=10, FAISS_THRESHOLD=0.68, CONFIDENCE_THRESHOLD=0.005
Recall@3=1.0000, MRR=0.9271, HitRate=1.0000, Composite=0.9781
```

> [!IMPORTANT]
> `RRF_K=10` (from the default 60) gives significantly more weight to top-ranked results.
> With k=60, rank-1 contributes `1/61 = 0.0164`. With k=10, rank-1 contributes `1/11 = 0.0909` — a **5.5× stronger signal** for the top result. This improves MRR because correctly-placed top results get amplified in the fused ranking.

---

## Applied Configuration

| Parameter | Module | New Value | Rationale |
|-----------|--------|-----------|-----------|
| `CONFIDENCE_THRESHOLD` | `vector_search.py` | `0.5648` | Midpoint of gap (0.5571, 0.5725) |
| `CONFIDENCE_THRESHOLD` | `bm25_search.py` | `0.7544` | Tuner recommended (midpoint of gap) |
| `RRF_K` | `hybrid_search.py` | `10` | Grid sweep winner — amplifies top results |
| `FAISS_THRESHOLD` | `hybrid_search.py` | `0.68` | Grid sweep winner — aggressive pre-fusion quality gate |
| `CONFIDENCE_THRESHOLD` | `hybrid_search.py` | `0.005` | OOS handled by FAISS_THRESHOLD; low bar for fused scores |

### Final Eval Results

```
Strategy   | Recall@3   | MRR        | Hit Rate  
--------------------------------------------------
BM25       | 1.0000     | 0.8646     | 1.0000    
Vector     | 1.0000     | 0.9062     | 1.0000    
Hybrid     | 1.0000     | 0.9271     | 1.0000    
```

> [!WARNING]
> These thresholds are tuned against a 16-query eval set. The vector search gap is only **0.0154** wide — a single adversarial query could close it. Expand the eval set (especially more OOS and near-miss queries) before treating these values as production-ready.

## Usage

Re-run the tuner anytime the eval set, embedding model, or corpus changes:

```bash
uv run python evals/tune_thresholds.py --k 3
```
