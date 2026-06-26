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

- [ ] Re-evaluate all saved answers for LC, Simple RAG, and Advanced RAG against
  `gold_paragraphs`.
- [ ] Include Phase 1, Phase 2, NIAH, and ablation outputs unless a report
  explicitly excludes one of these collections.
- [ ] Before including a collection, verify that every record has non-empty
  `gold_paragraphs`. This has already been checked for the active NIAH files,
  but the check should remain in the execution script.
- [ ] Use the same judge configuration across all systems.
- [ ] Save new JSONL files with a clear suffix, for example:
  `*_goldfaith.jsonl`.
- [ ] Label the metric as `gold_evidence_faithfulness`, not as the original
  paper metric.
- [ ] Recompute summary tables for the main Phase 1, Phase 2, NIAH, and ablation
  collections, then compare rankings against the published faithfulness tables.

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
- Paid smoke test has not been run.

Expected cost:

- LC only: under USD 1.
- All main systems: roughly USD 1-3.

Interpretation:

- This is a corrected robustness reanalysis.
- It evaluates answer grounding against human-verified evidence.
- It removes retrieval/attention failure from the comparison, so it is not a
  drop-in replacement for the original paper metric.

### Stage 3 - Decision Gate

- [ ] If Simple RAG still clearly beats LC under gold-evidence faithfulness,
  prepare a correction note stating that the magnitude changed but the ranking
  remains stable.
- [ ] If LC becomes close to or better than Simple RAG, treat the original
  faithfulness ranking as uncertain and move to Stage 4.
- [ ] Independently preserve the cost, quota-failure, and operational reliability
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
- [ ] Choose and document one asymmetry-handling policy before reporting:
  either rerun RAG faithfulness with its full retrieved context under the same
  uncapped evaluator, or label Option C narrowly as "LC evaluated against full
  generation context; RAG results remain retrieved-context evaluations."

Expected cost:

- Gemini 2.5 Flash Batch: roughly USD 12-25.
- Gemini 2.5 Flash standard: roughly USD 25-40.
- Premium Gemini 3 judge: potentially USD 50-100+.

Interpretation:

- This is the closest direct correction of the original automated LC
  faithfulness metric.
- It should be run only if Stage 2 materially changes the story or if a formal
  correction requires the closest possible reconstruction of the original metric.

### Stage 5 - Reporting

- [ ] State clearly that the published LC faithfulness numbers were affected by
  an evaluation-context logging bug.
- [ ] Distinguish `gold_evidence_faithfulness` from the original faithfulness
  metric.
- [ ] Report whether the corrected main Phase 1 and Phase 2 rankings change.
- [ ] Report whether the corrected NIAH ranking changes, if NIAH is included.
- [ ] Report whether the corrected ablation ranking changes, if ablation is
  included.
- [ ] Report whether the corrected effect magnitude changes.
- [ ] Keep the operational reliability finding separate:
  Phase 1 LC had 51 quota failures, all in the long-verdict stratum.
- [ ] Keep cost comparisons separate from corrected faithfulness claims.

## Human Spot-Check Framing

The human legal accuracy spot-check compared generated answers against gold
answers. It did not show the full source verdict to the reviewer.

The mismatch remains important: LC scored highest on human legal accuracy
(`1.80`) while automated LC faithfulness was much lower. This mismatch is
consistent with the evaluation bug, but it should not be described as evidence
from reviewers reading the full verdict text.

## Recommended Immediate Path

1. Complete Stage 1.
2. Run Stage 2 for all main systems.
3. Inspect the corrected ranking.
4. Decide whether Stage 4 is necessary.
5. Draft a transparent correction note only after seeing the Stage 2 numbers.
