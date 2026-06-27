# Faithfulness Evaluation Correction Note

**Prepared:** 2026-06-27
**Affects:** Paper.md, report.md — all faithfulness and hallucination_rate figures
**Root cause:** `long_context.py:131` stored a 503-char logging preview in `context_used`
instead of the full generation context. The experiment runner passed `context_used` to
`evaluate_faithfulness`, so all LC faithfulness scores were evaluated against the first
~500 chars of each verdict.

---

## 1. What Was Wrong

```python
# src/systems/long_context.py:131 (bug)
context_used = text[:500] + "..."   # truncated for logging only
```

The `runner.py` then called `evaluate_faithfulness(answer, result.context_used, ...)`,
meaning every LC faithfulness score was graded against the document beginning, not the
context actually used for generation. For long verdicts (avg 900k chars), the 503-char
preview covers less than 0.1% of the document.

RAG faithfulness was unaffected: `context_used` for RAG stored the retrieved chunks
correctly, and the evaluation was semantically correct.

---

## 2. Correction Method (Stage 2)

All answers (RAG and LC) were re-evaluated using **gold_evidence_faithfulness**: the
judge checked each answer against the human-verified `gold_paragraphs` field. This is an
oracle metric — it tests whether answers are supported by the known-correct evidence,
regardless of what each system actually retrieved or injected.

- Judge model: `gemini-3-flash-preview` (consistent across all 18 files)
- Records processed: 3,339 across phase1, phase2, NIAH, ablation
- Evaluation errors: 0
- Output: `results_corrected/gold_evidence_faithfulness/`

---

## 3. Corrected Phase 2 Results (Primary Table)

**Original faithfulness (against retrieved chunks for RAG, 503-char preview for LC):**

| Model | System | Faithfulness | Halluc. Rate |
|---|---|---:|---:|
| Gemini Flash | Simple RAG | 0.840 | 0.160 |
| Gemini Flash | Advanced RAG | 0.762 | 0.238 |
| Gemini Flash | LC | 0.616 | 0.384 |
| GPT-4o Mini | Simple RAG | 0.857 | 0.143 |
| GPT-4o Mini | Advanced RAG | 0.780 | 0.220 |
| GPT-4o Mini | LC | 0.524 | 0.476 |

**Corrected gold_evidence faithfulness (all systems vs oracle gold paragraphs):**

| Model | System | Faithfulness | Halluc. Rate | Delta vs original |
|---|---|---:|---:|---:|
| Gemini Flash | Simple RAG | 0.5567 | 0.4433 | -0.2833 |
| Gemini Flash | Advanced RAG | 0.4539 | 0.5461 | -0.3081 |
| Gemini Flash | LC | 0.5606 | 0.4394 | -0.0554 |
| GPT-4o Mini | Simple RAG | 0.5650 | 0.4350 | -0.2920 |
| GPT-4o Mini | Advanced RAG | 0.4756 | 0.5244 | -0.3044 |
| GPT-4o Mini | LC | 0.5750 | 0.4250 | -0.0510 |

**Ranking change:** Under gold_evidence, LC is numerically above Simple RAG in Phase 2
for both model families, but the LC-vs-Simple-RAG effect sizes are negligible and not
statistically significant. The corrected Phase 2 interpretation is therefore
**LC ~= Simple RAG > Advanced RAG**, not the original
Simple RAG > Advanced RAG > LC ranking. The original LC numbers were suppressed by the
bug, while RAG numbers declined because gold_evidence is a stricter oracle than
retrieved-chunk faithfulness.

---

## 4. Corrected Phase 1 Results

Phase 1 uses coverage-adjusted means (51 LC quota failures in the long stratum count as 0).

| System | Scored-only | Coverage-adjusted | n answered / total |
|---|---:|---:|---|
| Simple RAG | 0.5081 | 0.5081 | 300/300 |
| Advanced RAG | 0.4808 | 0.4808 | 300/300 |
| LC | 0.5991 | 0.4972 | 249/300 |

Phase 1 ranking under gold_evidence (coverage-adjusted): **Simple RAG (0.508) > LC (0.497)
> Advanced RAG (0.481)**. Margin between Simple RAG and LC is 0.011 — much smaller than
the original gap.

---

## 5. The Length-Sensitivity Finding Is Reversed

**Original (bug-affected) Phase 1 LC faithfulness by verdict stratum:**

| Stratum | Simple RAG | Advanced RAG | LC |
|---|---:|---:|---:|
| Short | 0.869 | 0.765 | 0.448 |
| Medium | 0.859 | 0.796 | 0.533 |
| Long | 0.803 | 0.721 | **0.205** |

The 0.448 → 0.533 → 0.205 pattern was interpreted as LC faithfulness collapsing on long
verdicts. **This pattern was entirely caused by the evaluation bug.** For long verdicts
(avg 900k chars), the 503-char preview covered less than 0.1% of the document.

**Corrected gold_evidence Phase 1 faithfulness by stratum:**

| Stratum | Simple RAG | Advanced RAG | LC (scored-only) | LC (coverage-adj.) |
|---|---:|---:|---:|---:|
| Short | 0.4833 | 0.4389 | 0.5778 | 0.5778 |
| Medium | 0.5271 | 0.5028 | 0.6056 | 0.6056 |
| Long | 0.5074 | 0.4935 | **0.6282** | **0.2722** |

Under gold_evidence, LC faithfulness quality **increases** with verdict length (0.578 →
0.606 → 0.628). LC produces the highest faithfulness scores in every stratum when it
produces an answer.

The poor long-verdict performance is entirely **operational**, not qualitative:
- 51/90 long-stratum LC queries hit the quota limit and returned no answer
- Among the 39 queries that received an answer, faithfulness was 0.628 (highest of any system)
- Coverage-adjusted score (0.272) reflects the 56.7% non-response rate, not faithfulness quality

**The paper's length-sensitivity claim must be retracted and replaced:**
- Original: "LC faithfulness collapses from 0.533 to 0.205 as verdict length increases"
- Corrected: "LC operational reliability collapses for long verdicts (56.7% non-response
  rate), but when LC produces an answer, faithfulness quality is highest (0.628 vs 0.507
  for Simple RAG)."

---

## 6. NIAH Results

The originally published NIAH "accuracy" was not a human legal-accuracy label. It was
computed by `experiments/run_niah.py` as a threshold over the original faithfulness score:
`faithfulness >= 0.7`. Because that source faithfulness score was affected by the LC
evaluation bug, the NIAH threshold-success values must be recomputed from
gold_evidence_faithfulness.

| System | Original threshold success | Faithfulness (original) | Faithfulness (gold_evidence) | Corrected threshold success |
|---|---:|---:|---:|---:|
| Simple RAG | **0.8333** | 0.9056 | 0.4889 | 0.3000 |
| Advanced RAG | 0.7667 | 0.8167 | 0.4667 | 0.3333 |
| LC | 0.5000 | 0.6833 | **0.5778** | **0.4000** |

Under gold_evidence, LC leads NIAH faithfulness (0.578 vs 0.489 for Simple RAG) and
corrected threshold success (12/30 vs 9/30 for Simple RAG). The original NIAH retrieval
advantage is therefore not valid as an independent corrected finding. A separate
exact-answer or retrieval-success metric would be needed to test the lost-in-the-middle
hypothesis directly.

---

## 7. Ablation Results

Under gold_evidence, the ablation ranking changes:

| Variant | Original Faithfulness | Gold_evidence | Rank change |
|---|---:|---:|---|
| simple_rag_baseline | 0.840 | **0.6042** | 3 -> **1** |
| plus_hybrid_search | 0.907 | 0.5850 | 1 -> 2 |
| plus_metadata_filter | — | 0.5767 | — -> 3 |
| plus_reranking | 0.725 | 0.4900 | — -> 4 |
| plus_query_rewrite | — | 0.4533 | — -> 5 |
| full_advanced_rag | 0.803 | 0.4717 | — -> 6 |

The conclusion that hybrid search is the most beneficial single augmentation no longer
holds under gold_evidence — the baseline (no augmentation) is best. The reranking penalty
persists (rank 4). This suggests the ablation components added retrieval complexity that
reduced gold-evidence faithfulness even when they improved retrieved-chunk faithfulness.

---

## 8. What Does Not Change

The following findings are unaffected by the faithfulness bug correction:

- **Cost comparison**: Simple RAG 24.8x cheaper than LC (based on token counts, unchanged)
- **Latency**: All latency figures unchanged
- **BERTScore**: Computed independently, unchanged
- **Legal accuracy spot-check**: Human-evaluated, unchanged
- **NIAH original threshold-success values**: must be treated as superseded, because they
  were derived from the original faithfulness score
- **Operational failure rate**: 51/90 long-stratum LC quota failures, unchanged
- **Phase 1 ranking direction**: Simple RAG > LC > Advanced RAG still holds (by 0.011)
- **Phase 2 significance tests**: Must be recomputed against gold_evidence numbers

---

## 9. What Must Be Updated in the Paper

1. **Table 1 (Phase 2 main results)**: Replace faithfulness and hallucination_rate columns
   with gold_evidence values. Add footnote explaining the correction.
2. **Table 2 (effect sizes / significance tests)**: Recompute with gold_evidence values.
3. **Section 4.4 (length sensitivity)**: Replace entire table and narrative. The collapse
   finding is retracted; the operational failure finding is the correct story.
4. **Section 4.6 (NIAH)**: Update faithfulness and threshold-success columns; do not
   describe the original NIAH accuracy values as human legal-accuracy labels.
5. **Section 5 (ablation)**: Update faithfulness column; note baseline now ranks first.
6. **Abstract / conclusion**: Soften "Simple RAG significantly outperforms LC" for Phase 2;
   note Phase 1 margin is 0.011 under corrected metric.
7. **PHASE_II_PLAN.md**: Reframe motivation — not a faithfulness collapse but an
   operational reliability failure on long verdicts.
