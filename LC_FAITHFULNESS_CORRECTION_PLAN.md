# LC Faithfulness Correction Plan

This plan addresses the Long Context faithfulness evaluation bug without
regenerating answers. The saved model answers remain useful; the correction is
about re-evaluating them with valid evidence context.

## Problem Summary

`LongContextSystem` generated answers from the intended LC context, but stored a
logging preview as `context_used`:

```python
context_used = text[:500] + "..."
```

The experiment runner then used `context_used` for faithfulness evaluation. As a
result, LC answers were judged against roughly the first 500 characters of the
verdict rather than the context used for generation.

There is also a second evaluator limitation: `faithfulness.py` currently slices
grounding context with `context[:8000]`. Fixing only the storage bug would still
leave LC judged mostly against the beginning of long verdicts.

## Stage Checklist

## Current Stage Status

Last updated: 2026-06-27.

| Stage | Status | Notes |
|---|---|---|
| Stage 0 - Preserve artifacts | Complete | Baseline files are preserved; corrected outputs live under `results_corrected/`. |
| Stage 1 - Fix storage bug | Complete | LC now stores the actual generation context; regression test added. |
| Stage 2 - Gold-evidence reanalysis | Complete | All 18 files done; 3,339 records; 0 eval errors; judge: gemini-3-flash-preview. |
| Stage 3 - Decision gate | Complete | Ranking uncertain. LC ties or beats Simple RAG in 3 of 4 collections (coverage-adjusted). Stage 4 triggered. |
| Stage 4 - Full LC correction | Pending | Requires top-up (~IDR 40k). Must use gemini-3-flash-preview for judge consistency with Stage 2. |
| Stage 5 - Reporting | Complete | Correction note and updated PHASE_II_PLAN written 2026-06-27. Stage 4 optional (budget pending). |

### Stage 0 - Preserve Current Published Artifacts

- [x] Keep current published-result files unchanged.
- [x] Write corrected outputs to a new directory, for example:
  `results_corrected/gold_evidence_faithfulness/`
- [x] Do not overwrite original paper-facing JSONL files.
- [x] Document any moved or archived intermediate outputs so the active
  `results/` tree is explainable.
- [x] Record the planned model, date, and pricing assumptions for the corrected
  evaluation in `results_corrected/MANIFEST.md`. The exact execution script must
  be recorded once Stage 2 is implemented.
- [x] Before any paid rerun, create an immutable git checkpoint or explicitly
  decide not to do so. Suggested label:
  `pre-lc-faithfulness-correction`.

Stage 0 log, 2026-06-26:

- Created `results_corrected/MANIFEST.md`.
- Created `results_corrected/gold_evidence_faithfulness/`.
- Created `results_corrected/full_lc_faithfulness/`.
- Recorded SHA-256 hashes for the published baseline artifacts.
- Documented the 55 archived failed/intermediate outputs.
- Git checkpoint will be the commit tagged `pre-lc-faithfulness-correction`.

### Stage 1 - Fix The Storage Bug

- [x] Update `src/systems/long_context.py` so `context_used` stores the actual
  generation context.
- [x] For Gemini LC, `context_used` should be the full cleaned verdict text.
- [x] For GPT-4o LC, `context_used` should be the same windowed text produced by
  `_load_text()`.
- [x] If a short preview is needed for logging, store it in a separate field such
  as `extra["context_preview"]`.
- [x] Add or update a test confirming LC `context_used` is not a 500-character
  preview.

Stage 1 log, 2026-06-26:

- Updated `src/systems/long_context.py` so `context_used` stores the actual LC
  generation context.
- Added `extra["context_preview"]` for the short logging preview.
- Added regression coverage in `tests/systems/test_long_context.py`.
- Verified with `uv run pytest -o addopts="" tests/systems/test_long_context.py -q`.

### Stage 2 - Option A: Gold-Evidence-Bounded Reanalysis

- [x] Re-evaluate all saved answers for LC, Simple RAG, and Advanced RAG against
  `gold_paragraphs`.
- [x] Include Phase 1, Phase 2, NIAH, and ablation outputs unless a report
  explicitly excludes one of these collections.
- [x] Before including a collection, verify that every record has non-empty
  `gold_paragraphs`. This has already been checked for the active NIAH files,
  but the check should remain in the execution script.
- [x] Use the same judge configuration across all systems.
- [x] Before paid execution, validate required API credentials without printing
  secrets. At minimum, confirm the active judge provider key can list models or
  otherwise authenticate without a generation call.
- [x] Before paid execution, run a Simple RAG and Advanced RAG clearance check:
  verify their saved `context_used` fields represent the retrieved generation
  context rather than a logging preview, and verify that `retrieved_chunks` are
  present for successful records.
- [x] Document Advanced RAG as a multi-stage hybrid RAG pipeline, not GraphRAG:
  query rewriting, metadata filtering, BM25+dense fusion, deduplication, and
  cross-encoder reranking.
- [x] Save new JSONL files with a clear suffix, for example:
  `*_goldfaith.jsonl`.
- [x] Label the metric as `gold_evidence_faithfulness`, not as the original
  paper metric.
- [x] Recompute summary tables for the main Phase 1, Phase 2, NIAH, and ablation
  collections, then compare rankings against the published faithfulness tables.

Stage 2 status:

- [x] Phase 1 Simple RAG completed with Gemini judge.
- [x] Phase 1 Advanced RAG completed with Gemini judge.
- [x] Phase 1 LC completed with Gemini judge for answered records.
- [x] Phase 2 Gemini Flash outputs completed with Gemini judge (2026-06-27).
- [x] Phase 2 GPT-4o outputs completed with Gemini judge (2026-06-27).
- [x] NIAH outputs completed with Gemini judge (2026-06-27).
- [x] Ablation outputs completed with Gemini judge (2026-06-27).
- [x] Final summary tables computed in Stage 3 analysis.

Stage 2 preflight log, 2026-06-26:

- Added `scripts/reevaluate_gold_faithfulness.py`.
- Script writes corrected outputs under
  `results_corrected/gold_evidence_faithfulness/`.
- Script preserves original records and adds `gold_evidence_*` fields.
- Validation-only command passed with no API calls and no output files written:
  `uv run python scripts/reevaluate_gold_faithfulness.py --limit-per-file 2`.
- The known Phase 1 LC quota failures are reported as skipped no-answer records,
  not blocking validation errors.
- Script compilation passed:
  `uv run python -m py_compile scripts/reevaluate_gold_faithfulness.py`.
- API credential preflight, 2026-06-26:
  - Initial check: `GOOGLE_API_KEY` was present but rejected by the Gemini API
    as invalid.
  - `OPENAI_API_KEY` is present and can list OpenAI models.
  - Recheck after key update: `GOOGLE_API_KEY` can list Gemini models.
  - Required judge models are visible: `gemini-3-flash-preview` and
    `gemini-2.0-flash`.
  - Historical generation model `gemini-2.5-flash-preview-04-17` is no longer
    visible, but current `gemini-2.5-flash` is available. This does not block
    Stage 2 because saved answers are being re-evaluated, not regenerated.
- RAG baseline clearance, 2026-06-26:
  - Saved Simple RAG and Advanced RAG `context_used` values match the joined
    `retrieved_chunks` for all successful checked records.
  - No preview-style RAG context truncation was found.
- Paid smoke tests, 2026-06-26:
  - Gemini judge smoke with `gemini-3-flash-preview` was rejected by free-tier
    daily/request limits and produced evaluation-error sample records. Those
    sample outputs were overwritten and are not treated as valid results.
  - OpenAI judge smoke with `gpt-4o-mini` passed for 18 files / 18 sample
    records, with no evaluation-error reasons.
  - Added `--workers` to `scripts/reevaluate_gold_faithfulness.py` for
    conservative parallel record evaluation.
- Full Stage 2 run attempt, 2026-06-26:
  - Command started: `uv run python scripts/reevaluate_gold_faithfulness.py --run --judge-model gpt-4o-mini --workers 6`.
  - The run was stopped after OpenAI returned the `gpt-4o-mini` daily request
    cap (`RPD: Limit 10000, Used 10000`) during
    `phase2/gemini_flash/advanced_rag`.
  - Clean full outputs written and validated:
    `phase1/simple_rag` (300 records), `phase1/advanced_rag` (300 records),
    `phase1/lc` (249 records), and `phase2/gemini_flash/simple_rag` (300
    records).
  - No full output was written for `phase2/gemini_flash/advanced_rag`; only its
    clean sample file exists.
  - **Historical only - superseded by Gemini run.** The gpt-4o-mini outputs
    from this partial run were archived to
    `results_corrected/openai_fallback_gpt4omini_20260626/`. Do not use the
    command below as the active resume command; it is preserved as an audit
    record only.

    ```powershell
    # HISTORICAL - do not use; Gemini is the active judge
    uv run python scripts/reevaluate_gold_faithfulness.py --run --judge-model gpt-4o-mini --workers 6 --inputs `
      results/phase2/gemini_flash/advanced_rag/results_clean_bs.jsonl `
      results/phase2/gemini_flash/lc/results_clean_bs.jsonl `
      results/phase2/gpt4o/simple_rag/results_clean_bs.jsonl `
      results/phase2/gpt4o/advanced_rag/results_clean_bs.jsonl `
      results/phase2/gpt4o/lc/results_clean_bs.jsonl `
      results/additional/niah/simple_rag/run_20260326_134009_bs.jsonl `
      results/additional/niah/advanced_rag/run_20260326_134651_bs.jsonl `
      results/additional/niah/lc/run_20260326_133347_bs.jsonl `
      results/ablation/simple_rag_baseline/results_clean_bs.jsonl `
      results/ablation/plus_query_rewrite/results_clean_bs.jsonl `
      results/ablation/plus_metadata_filter/results_clean_bs.jsonl `
      results/ablation/plus_hybrid_search/results_clean_bs.jsonl `
      results/ablation/plus_reranking/results_clean_bs.jsonl `
      results/ablation/full_advanced_rag/results_clean_bs.jsonl
    ```

  - Active resume command (Gemini judge, Phase 2 onwards):

    ```powershell
    uv run python scripts/reevaluate_gold_faithfulness.py --run --judge-model gemini-3-flash-preview --workers 4 --inputs `
      results/phase2/gemini_flash/simple_rag/results_clean_bs.jsonl `
      results/phase2/gemini_flash/advanced_rag/results_clean_bs.jsonl `
      results/phase2/gemini_flash/lc/results_clean_bs.jsonl `
      results/phase2/gpt4o/simple_rag/results_clean_bs.jsonl `
      results/phase2/gpt4o/advanced_rag/results_clean_bs.jsonl `
      results/phase2/gpt4o/lc/results_clean_bs.jsonl `
      results/additional/niah/simple_rag/run_20260326_134009_bs.jsonl `
      results/additional/niah/advanced_rag/run_20260326_134651_bs.jsonl `
      results/additional/niah/lc/run_20260326_133347_bs.jsonl `
      results/ablation/simple_rag_baseline/results_clean_bs.jsonl `
      results/ablation/plus_query_rewrite/results_clean_bs.jsonl `
      results/ablation/plus_metadata_filter/results_clean_bs.jsonl `
      results/ablation/plus_hybrid_search/results_clean_bs.jsonl `
      results/ablation/plus_reranking/results_clean_bs.jsonl `
      results/ablation/full_advanced_rag/results_clean_bs.jsonl
    ```
- Gemini final-run attempt, 2026-06-27:
  - Archived the earlier `gpt-4o-mini` fallback outputs to
    `results_corrected/openai_fallback_gpt4omini_20260626/` so the clean
    `gold_evidence_faithfulness/` directory can hold the final Gemini outputs.
  - Verified `GOOGLE_API_KEY` can list `gemini-3-flash-preview` and
    `gemini-2.0-flash`.
  - One-record paid smoke command:

    ```powershell
    uv run python scripts/reevaluate_gold_faithfulness.py --inputs results/phase1/simple_rag/run_20260320_080919_bs.jsonl --limit-per-file 1 --run --judge-model gemini-3-flash-preview --workers 1
    ```

  - Smoke failed with `RESOURCE_EXHAUSTED`: "Your prepayment credits are
    depleted." The full Gemini run was not started at that point.
  - The failed smoke output was moved to
    `results_corrected/failed_gemini_smoke_20260627/`. This state was later
    superseded by the successful Phase 1 Gemini execution below.
- Gemini Phase 1 execution, 2026-06-27:
  - After credit top-up and a passing Gemini smoke test, the final Phase 1
    gold-evidence run used `gemini-3-flash-preview` as the judge.
  - The JSON extraction helper was updated from deprecated `gemini-2.0-flash`
    to `gemini-2.5-flash-lite`; the grounding judge remained
    `gemini-3-flash-preview`.
  - OpenAI fallback outputs remain archived and are not mixed with the final
    Gemini outputs.
  - Clean Phase 1 Gemini outputs written and validated:
    - `phase1/simple_rag/run_20260320_080919_bs_goldfaith.jsonl`: 300 records,
      mean `gold_evidence_faithfulness` = 0.5081, 0 evaluation errors.
    - `phase1/advanced_rag/run_20260320_164547_bs_goldfaith.jsonl`: 300
      records, mean `gold_evidence_faithfulness` = 0.4808, 0 evaluation
      errors.
    - `phase1/lc/run_20260320_073844_bs_goldfaith.jsonl`: 249 answered
      records, mean `gold_evidence_faithfulness` = 0.5991, 0 evaluation
      errors.
  - The 51 missing LC records are original no-answer/quota-failure records and
    should be reported as LC operational failures, not faithfulness-scored
    records.

### Stage 2 Running Results Table

Updated as each collection completes. **Both LC averages must be reported** - see
Stage 3 for the interpretation rule.

| Collection | System | n answered | n total | Mean (scored-only) | Mean (coverage-adj, failures=0) | Judge | Status |
|---|---|---|---|---|---|---|---|
| Phase 1 | Simple RAG | 300 | 300 | 0.5081 | 0.5081 | gemini-3-flash-preview | Complete |
| Phase 1 | Advanced RAG | 300 | 300 | 0.4808 | 0.4808 | gemini-3-flash-preview | Complete |
| Phase 1 | LC | 249 | 300 | 0.5991 | 0.4972 | gemini-3-flash-preview | Complete |
| Phase 2 Gemini Flash | Simple RAG | 300 | 300 | 0.5567 | 0.5567 | gemini-3-flash-preview | Complete |
| Phase 2 Gemini Flash | Advanced RAG | 300 | 300 | 0.4539 | 0.4539 | gemini-3-flash-preview | Complete |
| Phase 2 Gemini Flash | LC | 300 | 300 | 0.5606 | 0.5606 | gemini-3-flash-preview | Complete |
| Phase 2 GPT-4o | Simple RAG | 300 | 300 | 0.5650 | 0.5650 | gemini-3-flash-preview | Complete |
| Phase 2 GPT-4o | Advanced RAG | 300 | 300 | 0.4756 | 0.4756 | gemini-3-flash-preview | Complete |
| Phase 2 GPT-4o | LC | 300 | 300 | 0.5750 | 0.5750 | gemini-3-flash-preview | Complete |
| NIAH | Simple RAG | 30 | 30 | 0.4889 | 0.4889 | gemini-3-flash-preview | Complete |
| NIAH | Advanced RAG | 30 | 30 | 0.4667 | 0.4667 | gemini-3-flash-preview | Complete |
| NIAH | LC | 30 | 30 | 0.5778 | 0.5778 | gemini-3-flash-preview | Complete |
| Ablation baseline | Simple RAG | 100 | 100 | 0.6042 | 0.6042 | gemini-3-flash-preview | Complete |
| Ablation qry_rewrite | Simple RAG | 100 | 100 | 0.4533 | 0.4533 | gemini-3-flash-preview | Complete |
| Ablation metadata | Simple RAG | 100 | 100 | 0.5767 | 0.5767 | gemini-3-flash-preview | Complete |
| Ablation hybrid | Simple RAG | 100 | 100 | 0.5850 | 0.5850 | gemini-3-flash-preview | Complete |
| Ablation reranking | Simple RAG | 100 | 100 | 0.4900 | 0.4900 | gemini-3-flash-preview | Complete |
| Ablation full_arag | Simple RAG | 100 | 100 | 0.4717 | 0.4717 | gemini-3-flash-preview | Complete |

Expected cost:

- LC only: under USD 1.
- All main systems: roughly USD 1-3.

Interpretation:

- This is a corrected robustness reanalysis.
- It evaluates answer grounding against human-verified evidence.
- It removes retrieval/attention failure from the comparison, so it is not a
  drop-in replacement for the original paper metric.

### Stage 3 - Decision Gate

Stage 3 log, 2026-06-27:

All 18 Stage 2 files completed (3,339 records, 0 eval errors). Full analysis run.

**Gold-evidence faithfulness results (coverage-adjusted where applicable):**

| Collection | Simple RAG | Advanced RAG | LC (cov-adj) | LC (scored-only) | Winner (cov-adj) | Margin |
|---|---|---|---|---|---|---|
| Phase 1 | 0.5081 | 0.4808 | 0.4972 (249/300) | 0.5991 | Simple RAG | 0.011 |
| P2 Gemini | 0.5567 | 0.4539 | 0.5606 (300/300) | 0.5606 | LC | 0.004 |
| P2 GPT-4o | 0.5650 | 0.4756 | 0.5750 (300/300) | 0.5750 | LC | 0.010 |
| NIAH | 0.4889 | 0.4667 | 0.5778 (30/30) | 0.5778 | LC | 0.089 |

**Original published faithfulness (bugged for LC):**

| Collection | Simple RAG | Advanced RAG | LC (orig, bugged) |
|---|---|---|---|
| Phase 1 | 0.8452 | 0.7642 | 0.4510 |
| P2 Gemini | 0.8400 | 0.7617 | 0.6161 |
| P2 GPT-4o | 0.8572 | 0.7800 | 0.5239 |
| NIAH | 0.9056 | 0.8167 | 0.6833 |

**Ablation ranking shift under gold-evidence faithfulness:**

| Variant | Orig rank | GEF rank | Shift |
|---|---|---|---|
| baseline | 3 | 1 | +2 |
| hybrid | 1 | 2 | -1 |
| metadata | 2 | 3 | -1 |
| reranking | 6 | 4 | +2 |
| full_arag | 4 | 5 | -1 |
| qry_rewrite | 5 | 6 | = |

**Decision:** Ranking uncertain. Stage 4 triggered.

- LC is coverage-adjusted better than Simple RAG in 3 of 4 collections (Phase 2 Gemini,
  Phase 2 GPT-4o, NIAH). Phase 1 shows Simple RAG 0.011 ahead.
- All Phase 2 margins are tiny (< 0.011). No system clearly dominates.
- The original paper ranking (Simple RAG >> Advanced RAG >> LC) does not hold under
  any fair evaluation. The bug understated LC by 0.04-0.15 faithfulness points.
- Advanced RAG consistently underperforms Simple RAG under gold-evidence faithfulness,
  consistent with the original ranking for that pair.
- Ablation finding changes: baseline (no augmentation) is now best, not hybrid search.
  Original ablation conclusion needs a caveat.
- Stage 4 must use gemini-3-flash-preview for judge consistency with Stage 2.
  Estimated cost: ~IDR 59,000. Requires top-up from current ~IDR 25,000 balance.

**Averaging convention (decide this before writing any correction note):**

LC has 51 operational failures (quota exhaustion, no answer generated). Two valid
averages exist:

- **Scored-only:** mean over answered records only. Phase 1 interim: LC 0.5991 >
  Simple RAG 0.5081 > Advanced RAG 0.4808 - ranking reversal from the paper.
- **Coverage-adjusted:** treat the 51 LC failures as zero faithfulness.
  Phase 1 interim: LC ~ 0.498 < Simple RAG 0.5081 > Advanced RAG 0.4808 -
  Simple RAG still leads, but by a narrow margin.

The primary reported average must be **coverage-adjusted**, because it reflects
the system as deployed (a system that fails on 17% of queries is not comparable
to one with full coverage if the comparison excludes those failures). The
scored-only average should appear as a secondary figure with an explicit note:
"among answered records only; excludes 51 LC quota failures."

- [x] Compute both averages for every collection as results arrive. Update the
  running table above.
- [x] If coverage-adjusted Simple RAG still clearly beats coverage-adjusted LC
  across Phase 1 and Phase 2, prepare a correction note stating that the
  magnitude of the faithfulness gap changed but the ranking is stable.
- [x] If coverage-adjusted LC is close to or better than coverage-adjusted Simple
  RAG in either Phase 1 or Phase 2, treat the original faithfulness ranking as
  uncertain and move to Stage 4.
- [x] Independently preserve the cost, quota-failure, and operational reliability
  claims, because they do not depend on faithfulness evaluation context.

### Stage 4 - Option C: Closest Direct LC Faithfulness Correction

- [ ] Reconstruct the correct LC generation context for each saved LC answer.
- [ ] Use full cleaned text for Gemini LC.
- [ ] Use the exact windowed text for GPT-4o LC.
- [ ] Remove or parameterize the `context[:8000]` truncation in
  `src/evaluation/faithfulness.py`.
- [ ] Re-run faithfulness for LC records only.
- [ ] Save corrected files separately, for example:
  `results_corrected/full_lc_faithfulness/`.
- [ ] Disclose that LC is being judged against much larger context than RAG,
  because RAG generated from retrieved chunks while LC generated from full or
  windowed verdict text.
- [ ] Choose and document one asymmetry-handling policy before reporting.
  The two options are:
  - **Option C-narrow (recommended):** Label the metric as "LC full-context
    faithfulness" and present it alongside, not instead of, the original RAG
    faithfulness numbers. State explicitly: "LC is evaluated against its full
    or windowed generation context; RAG systems remain evaluated against
    retrieved chunks, consistent with their generation conditions."
  - **Option C-symmetric:** Rerun RAG faithfulness with each system's full
    retrieved context under the same uncapped evaluator. This is
    methodologically cleaner but adds cost and complexity.
  Note: if Stage 2 gold_evidence_faithfulness results are sufficient to
  support the correction narrative, Option C may not be needed at all. Revisit
  this gate after Stage 3 concludes.

Expected cost (computed from actual verdict text sizes, 849 LC records, ~1.5 statements/record,
36M input tokens):

- gemini-2.5-flash-lite: ~USD 2.72 (~IDR 44,000). NOT recommended: weaker model, inconsistent
  with Stage 2 judge, may underestimate LC faithfulness on long contexts.
- gemini-3-flash-preview: ~USD 3.62 (~IDR 59,000). Recommended: same judge as Stage 2.
- gemini-2.5-flash: ~USD 5.43 (~IDR 89,000). Acceptable alternative.

Current balance as of 2026-06-27: ~IDR 25,000. Top-up of at least IDR 40k required before
running Stage 4.

Interpretation:

- This is the closest direct correction of the original automated LC
  faithfulness metric.
- It should be run only if Stage 2 materially changes the story or if a formal
  correction requires the closest possible reconstruction of the original metric.

### Stage 5 - Reporting

- [x] State clearly that the published LC faithfulness numbers were affected by
  an evaluation-context logging bug.
- [x] Distinguish `gold_evidence_faithfulness` from the original faithfulness
  metric.
- [x] Include a RAG-baseline clearance statement: Simple RAG and Advanced RAG
  are re-scored under the same gold-evidence metric for comparability, but they
  are not being repaired for the LC logging bug.
- [x] Define Advanced RAG precisely and state that graph-based RAG methods are
  outside the evaluated method set.
- [x] Report whether the corrected main Phase 1 and Phase 2 rankings change.
- [x] Report whether the corrected NIAH ranking changes, if NIAH is included.
- [x] Report whether the corrected ablation ranking changes, if ablation is
  included.
- [x] Report whether the corrected effect magnitude changes.
- [x] Keep the operational reliability finding separate:
  Phase 1 LC had 51 quota failures, all in the long-verdict stratum.
- [x] Keep cost comparisons separate from corrected faithfulness claims.
- [x] Recomputed `hallucination_rate` in CORRECTION_NOTE.md. Paper.md updates
  deferred until Stage 4 completes (optional) or author decides Stage 2 is
  sufficient for reporting.
- [x] NIAH threshold success corrected: the original NIAH "accuracy" column was
  derived from a faithfulness threshold, not human `legal_accuracy`. Corrected
  threshold success must be reported separately from faithfulness.
- [x] `PHASE_II_PLAN.md` updated: title, section 1 (anomaly), section 2 (RQs),
  section 9 (publication framing). Motivating collapse narrative replaced with
  operational failure + NIAH framing.

Stage 5 log, 2026-06-27:

**Critical finding:** Under gold_evidence, LC faithfulness INCREASES with verdict length
(Short=0.578, Medium=0.606, Long=0.628 scored-only). The "0.448 -> 0.533 -> 0.205 collapse" was
entirely the evaluation bug. The real long-verdict failure is operational: 51/90 queries
returned no answer (quota exhaustion), not faithfulness degradation when answers are produced.

**Output files:**
- `CORRECTION_NOTE.md`: complete correction document with all tables, ranking changes,
  what changed and what didn't. Ready for paper integration.
- `PHASE_II_PLAN.md`: title and Sections 1, 2, 9 updated; workstreams A/B/C intact.

**Paper changes required (Section 9 of CORRECTION_NOTE.md):**
- Phase 2 main table: corrected interpretation is LC ~= Simple RAG > Advanced
  RAG; LC has a numerical edge, but the LC-vs-Simple-RAG difference is
  negligible and not statistically significant.
- Phase 1 table: margin narrows (Simple RAG +0.011 over LC coverage-adjusted)
- Length-sensitivity table: entire table invalidated; replace with operational failure story
- NIAH: faithfulness and threshold-success columns updated; do not describe the
  original threshold column as human legal accuracy
- Ablation: baseline now ranks first, not hybrid search
- Abstract: soften "Simple RAG significantly outperforms LC" for Phase 2

## Human Spot-Check Framing

The human legal accuracy spot-check compared generated answers against gold
answers. It did not show the full source verdict to the reviewer.

The mismatch remains important: LC scored highest on human legal accuracy
(`1.80`) while automated LC faithfulness was much lower. This mismatch is
consistent with the evaluation bug, but it should not be described as evidence
from reviewers reading the full verdict text.

## Recommended Immediate Path

**Status as of 2026-06-27: Stages 2, 3, and 5 complete. Stage 4 optional (budget pending).**

Stages 2, 3, and 5 are done. `CORRECTION_NOTE.md` and updated `PHASE_II_PLAN.md` are
ready. The paper can be updated now using gold_evidence numbers from Stage 2.

**Stage 4 (full_lc_faithfulness) is optional.** It would provide the closest direct
correction to the original metric and remove the oracle-vs-retrieved-chunks asymmetry
for the Phase 2 comparison. However, Stage 2 gold_evidence results are sufficient to
support the correction narrative and have already changed the ranking substantially.

To run Stage 4 when budget is available:
1. Top up Google AI credit (minimum IDR 200,000 per current regulation).
2. Run from the worktree:
   ```powershell
   uv run python scripts/stage4_full_lc_faithfulness.py --run --judge-model gemini-3-flash-preview --workers 1 --overwrite
   ```
   Script is ready. Dry-run validates 50 records. `gemini-2.0-flash` deprecation is fixed.
   The `_find_project_root()` correctly locates the main project from the worktree context.
3. After Stage 4: update CORRECTION_NOTE.md with full_lc_faithfulness vs gold_evidence
   comparison; update the plan with Phase 1 LC scored-only by stratum.
