# VerdictBench Part II — Mechanistic Diagnosis of Long-Context Operational Failure

**Working title:** *Why Long Context Fails: Positional, Structural, and Mechanistic Analysis of LC Operational Failure on Indonesian Constitutional Verdicts*

**Status:** Pre-research planning document. Last updated: 2026-06-27.

> **CORRECTION (2026-06-27):** The Phase I figure cited as motivation for Phase II — "LC faithfulness collapses from 0.803 → 0.205 on long verdicts" — was produced by an evaluation bug. See `CORRECTION_NOTE.md` for full details. Under the corrected gold_evidence metric, LC faithfulness *increases* with verdict length (Short=0.578, Medium=0.606, Long=0.628 scored-only). The actual anomaly is **operational**: 51/90 (56.7%) of long-stratum LC queries returned no answer due to quota exhaustion. The Phase II research questions have been reframed accordingly. All three workstreams remain valid; only the primary motivating claim changes.

---

## 1. The Anomaly Worth Explaining

Phase I established the ranking. Phase II explains the mechanism behind the most striking set of numbers from Phase I — which, after the faithfulness evaluation correction, are:

1. **LC long-stratum non-response rate: 56.7%** (51/90 queries returned no answer due to quota exhaustion). Among the 39 that answered, faithfulness was 0.628 — the highest of any system. The failure is entirely operational.
2. **NIAH accuracy: LC 0.500 vs. Simple RAG 0.833.** LC correctly locates deep-evidence needles only half the time. This is based on human-evaluated `legal_accuracy` and is unaffected by the faithfulness correction.
3. **Phase I coverage-adjusted ranking: Simple RAG (0.508) > LC (0.497).** The margin is small (0.011) but the gap is explained by non-response, not by lower faithfulness quality when LC answers.

These three numbers triangulate the same failure mode: LC does not degrade *qualitatively* on long verdicts, it fails to *complete* the task. The question is whether that reflects a retrieval/attention failure (the model cannot find the answer in a long context) or a generation/quota failure (the model runs out of quota before finishing).

The Phase I data already contains enough structure to support mechanistic hypotheses. Before running new experiments, we should extract everything it tells us. Only then run targeted new experiments to fill the gaps.

The three diagnostic lenses remain:

1. **Positional**: Does LC non-response rate or faithfulness-when-answered vary as a function of gold-evidence depth within the document? (Tests whether "lost in the middle" explains the 56.7% non-response.)
2. **Mechanistic**: What are the internal attention and gradient patterns when the model fails to locate and use the answer passage? (Tests the "boilerplate distraction" hypothesis with attribution methods.)
3. **Taxonomic**: Among answered-but-unfaithful LC outputs, what failure mode dominates — hallucination, omission, or wrong-passage attendance? (Human-readable, directly actionable.)

---

## 2. Research Questions

**RQ1 (Positional):** Does the LC non-response rate (quota exhaustion / empty answer) vary as a function of gold-evidence depth? Does faithfulness-when-answered also vary with depth, or is the faithfulness quality uniform once LC produces an answer? This separates two distinct mechanisms: a *retrieval* failure (can't find the answer) vs. a *generation* failure (runs out of budget before finishing).

**RQ2 (Cross-type):** Does the positional effect differ by question type? Multi-section reasoning questions span multiple passages — the "lost in the middle" mechanism predicts higher non-response rates and lower faithfulness for these vs. factual extractive questions pointing to a single passage.

**RQ3 (Mechanistic):** On an open-weight multilingual model, can we identify specific attention heads or layers that distinguish grounded from ungrounded LC generations? When the model fails to retrieve the answer passage, is attention disproportionately allocated to lexically salient tokens (boilerplate legal headers, article numbers) rather than the answer-bearing content?

**RQ4 (Taxonomic):** Among answered-but-unfaithful LC outputs on long verdicts, what fraction fall into each failure mode — hallucination (confident confabulation), omission (partial but incomplete answer), wrong-passage attendance (plausible answer from wrong section), or truncation? Is the distribution different for factual vs. reasoning questions?

**RQ5 (NIAH mechanism):** Why does LC accuracy drop to 50% on NIAH deep-evidence queries (vs. 83.3% for Simple RAG)? Is this a depth effect (needle is in the document's deep middle) or a question-type effect (NIAH questions require locating a specific buried fact)?

---

## 3. Workstream A — Positional Faithfulness Analysis

### 3.1 Phase I Re-analysis (no new experiments required)

The existing `results/phase1/lc/run_20260320_073844_bs.jsonl` contains 249 scored LC records. Each record has:
- `gold_paragraphs`: list of strings — the exact evidence passage(s) the answer should be grounded in
- `verdict_id`: which of the 50 verdicts
- `faithfulness`: the label

**Procedure:**
1. For each scored LC record, load `data/processed/cleaned/{verdict_id}.txt`
2. Search for `gold_paragraphs[0]` within the cleaned text via `str.find()` (exact match) with a fallback to fuzzy match if OCR introduces spacing differences
3. Compute `depth = char_position / total_chars` ∈ [0, 1]
4. Plot faithfulness vs. depth; fit a LOWESS smoother and a linear regression
5. Run the same analysis stratified by question type (factual_extractive vs. multi_section_reasoning are the two largest categories)

**Expected output:** `notebooks/11_positional_analysis.ipynb` producing:
- Scatter plot: faithfulness vs. gold-evidence depth for 249 LC scored records
- Box plot: faithfulness by depth quartile (Q1 = 0–25%, Q2 = 25–50%, Q3 = 50–75%, Q4 = 75–100%)
- Stratified analysis by question type

**Limitation:** The existing data is observational — verdict length and evidence position are correlated. Long verdicts have higher token count AND are the ones that fail. A controlled positional experiment (Workstream A.2) separates these.

### 3.2 Controlled Positional Injection Experiment (new runs)

Construct synthetic LC prompts where the gold passage is placed at a controlled depth. For each of the 39 long-stratum LC questions that succeeded in Phase I (and whose gold paragraph we can isolate):

1. Take the full cleaned verdict text
2. Split it at the target depth
3. Reconstruct the context as: `prefix_text + SEPARATOR + gold_paragraph + SEPARATOR + suffix_text`
4. Run each question at 5 depth levels: 5%, 25%, 50%, 75%, 95%
5. Evaluate faithfulness for each placement

The SEPARATOR should be a generic legal marker (`\n\n---\n\n`) that doesn't signal position to the model. The total document length remains constant — only the position of the gold paragraph changes.

**Sample size:** 39 questions × 5 positions = 195 runs. With faithfulness evaluation, ~$1–2 total (Gemini Flash rates).

**Expected output:** `experiments/run_positional.py`, `results/additional/positional/`, `notebooks/11_positional_analysis.ipynb`

**Confound to control:** Some questions have `gold_paragraphs` of length > 1 (multi-section). For these, compute the mean position of all gold paragraphs.

**Risk:** Synthetic contexts may be detectable as artificial (the document structure changes). Mitigation: verify that the reconstructed context has the same total length and that the surrounding text is semantically unrelated to the answer.

---

## 4. Workstream B — Failure Mode Taxonomy

### 4.1 Sampling Strategy

From the Phase I LC results, filter to records where:
- `stratum == 'long'` (the collapse stratum)
- `faithfulness <= 0.4` (clear failures)

This gives approximately 25–30 records from the 39 long-stratum scored queries (those with faithfulness ≤ 0.4). For diversity, also sample 10–15 records from medium and short strata with faithfulness ≤ 0.4.

Target: ~45 records for the taxonomy.

### 4.2 Failure Mode Taxonomy

Each record should be independently read against the source verdict and classified into one primary failure mode:

| Code | Label | Definition |
|:---|:---|:---|
| `H` | **Hallucination** | Model generates confident, specific, factually wrong claims not present anywhere in the verdict (wrong names, dates, article numbers, invented legal reasoning) |
| `O` | **Omission** | Model gives a partial answer that is correct as far as it goes but fails to address a critical part of the question — identifiable by comparing to the gold paragraph |
| `WP` | **Wrong-passage attendance** | Model produces a plausible-sounding answer drawn from the wrong section of the verdict (recognizable because the content is real but not the answer to this question) |
| `TR` | **Truncation/failure to engage** | Model output is empty, cut off mid-sentence, or refuses to answer ("the document does not contain...") even though the answer is present |
| `BP` | **Boilerplate parroting** | Model reproduces a generic legal sentence (procedural formula common across many verdicts) without grounding in the specific evidential passage |

**Why this taxonomy is hard to automate:** `WP` and `H` look similar to the faithfulness judge (both are "unsupported") but have very different implications. `WP` suggests the model read the document but attended to the wrong section — fixable with better retrieval. `H` suggests the model ignored the document and drew from parametric knowledge — fixable with better grounding prompts or RAG. Distinguishing them requires reading Indonesian legal text and knowing what similar sections exist in the document.

### 4.3 Annotation Protocol

1. Annotator reads: the question, the gold paragraph(s), the model's answer, and the verdict section containing the gold paragraph (±500 chars context)
2. Assigns primary code + confidence (high/low)
3. Free-text note (1–2 sentences)
4. Second annotator reviews a 20% overlap set (9–10 records) using the same rubric

**Expected output:** `data/failure_taxonomy/lc_long_failures_annotated.jsonl`, `notebooks/12_failure_taxonomy.ipynb`

**Deliverable:** A pie chart / Sankey diagram of failure mode distribution, plus representative examples for each category.

---

## 5. Workstream C — Mechanistic Attention Analysis

### 5.1 Model Selection

The production experiments used Gemini 2.5 Flash (proprietary, no activation access). The mechanistic analysis requires an open-weight model with strong multilingual capabilities. Recommended:

**Primary:** `Qwen/Qwen2.5-7B-Instruct` — strong multilingual (Indonesian is in the training set), instruction-tuned, 128k context window, good transformer tooling support. 7B is large enough to surface real attention phenomena while being runnable on a single A100.

**Fallback:** `meta-llama/Llama-3.2-3B-Instruct` — smaller, faster iteration for tooling setup; weaker Indonesian but useful for initial calibration.

**Important caveat:** Findings from open-weight attention analysis are *suggestive* of a mechanism, not direct evidence that Gemini 2.5 Flash fails for the same reason. Frame findings as "a plausible mechanistic account" in the paper, not a causal explanation of the production model.

### 5.2 Analysis 1 — Attention Head Analysis

**Tool:** `bertviz` (for token-level attention) or `TransformerLens` (for head-level composition analysis).

**Procedure:**
1. Take 20 representative queries from the positional experiment (10 where LC succeeds, 10 where it fails) — matched on verdict length
2. For each: run the model in `output_attentions=True` mode
3. Aggregate attention from the generation tokens back to the input tokens
4. Compute two metrics per position in the input:
   - `attn_to_gold_paragraphs`: sum of attention weights over tokens in the gold passage
   - `attn_to_boilerplate`: sum over known boilerplate sections (formal court openings, signature blocks)
5. Compare distributions between success and failure cases

**Hypothesis (testable):** In failure cases, `attn_to_gold_paragraphs` is lower and `attn_to_boilerplate` is higher compared to success cases.

**Known limitation:** Attention weights are not a direct measure of importance (Jain & Wallace, 2019). Use alongside gradient-based attribution.

### 5.3 Analysis 2 — Gradient Attribution

**Tool:** `captum` (Integrated Gradients) or manual gradient × embedding computation.

**Procedure:**
1. For each of the 20 queries, compute Integrated Gradients with respect to the input embeddings, using the generated token sequence as the target
2. Aggregate IG scores over token positions
3. Compare the IG-derived "importance map" against the gold paragraph positions and the attention-derived maps from Analysis 1

If IG and attention agree on where importance mass lands, the attention story is more credible. If they diverge, rely on IG as the more causally grounded measure.

### 5.4 Analysis 3 — Logit Lens / Residual Stream Decomposition

**Tool:** TransformerLens `logit_lens` function.

**Procedure:**
1. For 5–10 key queries, trace what "answer" the model would generate if we read off the residual stream at each layer (0 through N)
2. For failures: at what layer does the answer "go wrong"? Does the model start with the correct answer and then corrupt it (failure of late-layer grounding) or never form the correct answer (failure of middle-layer retrieval)?
3. This distinguishes two hypotheses:
   - **Early failure**: the correct information was never retrieved from context
   - **Late corruption**: the correct information was present in the residual stream but was overwritten by parametric knowledge or long-range interference

### 5.5 Infrastructure Note

This workstream requires a GPU environment not in the current repo. Suggested setup:
- New script: `experiments/run_mechanistic.py` with `--model`, `--queries-file`, `--output-dir` flags
- Dependencies: `transformers`, `torch`, `bertviz`, `TransformerLens`, `captum` — add to a separate `requirements-interpretability.txt` so Phase I dependencies aren't affected
- Storage: activation tensors for 20 queries × 32 layers × 32 heads are large; save only aggregated statistics to disk, not raw activations

---

## 6. Proposed Experiment Script Layout

```
experiments/
  run_positional.py           # new: 5-depth controlled injection
  run_mechanistic.py          # new: open-weight attention/IG analysis
  run_failure_taxonomy.py     # new: filter + export failure records for human annotation

data/
  failure_taxonomy/
    lc_long_failures.jsonl    # auto-generated: filtered LC failures
    lc_long_failures_annotated.jsonl  # human-annotated

results/
  additional/
    positional/               # per-depth faithfulness results
    mechanistic/              # aggregated attention/IG statistics

notebooks/
  11_positional_analysis.ipynb
  12_failure_taxonomy.ipynb
  13_mechanistic_attention.ipynb
```

---

## 7. Methodological Risks and Mitigations

| Risk | Severity | Mitigation |
|:---|:---|:---|
| Open-weight model doesn't replicate Gemini's failure pattern | High | Frame as proxy analysis; prioritize Workstreams A and B as the primary evidence |
| Attention is not a reliable importance measure | Medium | Run IG analysis alongside attention; only claim mechanism if both agree |
| Synthetic positional prompts are detectable as artificial | Medium | Verify: generate 5 prompts per document and check response consistency |
| Gold paragraph search fails (OCR artifacts cause `str.find()` miss) | Low | Implement fuzzy matching with Levenshtein distance threshold ≤ 0.05 |
| Human taxonomy annotations require legal Indonesian reading | High | Annotator must be the same person who built the QA dataset; second-annotator overlap is calibration, not substitution |
| Indonesian legal text is rare in open-weight training data | Medium | Run baseline quality check: test Qwen2.5-7B-Instruct on 10 Phase I questions, verify faithfulness ≥ 0.4 before committing to full mechanistic analysis |

---

## 8. Expected Contributions

1. **Positional faithfulness curve for Indonesian legal LC.** A quantitative answer to "how much does position matter" — does 25% depth feel like 95% depth, or is the degradation gradual? This extends Liu et al. (2023) to a non-English, structured legal domain.

2. **First failure-mode taxonomy for LC on legal QA.** The distinction between hallucination, wrong-passage attendance, and omission is actionable — each has a different fix. This is directly useful for practitioners deploying LC systems on legal text.

3. **Mechanistic account of the collapse.** An attribution-level explanation that goes beyond "LC underperforms" to "here is what the model is attending to instead of the answer" — the most publishable contribution and the one most relevant to MATS / Anthropic Fellows reviewers.

4. **Reuse of existing infrastructure.** All three workstreams reuse the existing 300-QA benchmark, the faithfulness judge pipeline, and the `QASystem` interface — no new benchmark construction required.

---

## 9. Publication Positioning

**Target venue:** EMNLP 2026 main track or Findings, or *ACL 2027* with a longer experimental timeline.

**Framing:** Not "another RAG benchmark" but "mechanistic interpretability applied to a real-world IR-failure mode." The paper's hook is: Phase I found that LC *qualitatively* surpasses RAG at faithfulness when it answers (0.628 vs 0.507 for long verdicts), yet *operationally* fails 57% of the time on those same queries. Phase II explains the operational failure mechanism. That structure (paradox → mechanism → implication) is sharper than the original "collapse" framing and more publishable.

**Fellowship legibility:** The mechanistic attention analysis (Workstream C) directly demonstrates engagement with interpretability methods. The failure taxonomy (Workstream B) demonstrates human judgment in the loop — neither component can be replicated by running more API calls. This is the combination that distinguishes the work.

---

## 10. Suggested Implementation Order

| Priority | Workstream | Estimated effort | Depends on |
|:---|:---|:---|:---|
| 1 | A.1 — Phase I re-analysis | 2–3 days | Nothing new; uses existing data |
| 2 | B — Failure taxonomy | 3–5 days | Gold paragraph positions from A.1 (to select representative failures) |
| 3 | A.2 — Positional injection experiment | 4–6 days | A.1 findings to calibrate depth levels; API budget ~$2 |
| 4 | C.1 — Attention analysis | 1–2 weeks | GPU access; Qwen2.5-7B setup |
| 5 | C.2/C.3 — IG + logit lens | 1–2 weeks | C.1 tooling setup |

Start with A.1 because it costs nothing and may sharpen the hypotheses for B and C. Do B before C because the failure taxonomy will inform what "failure" looks like concretely — that context makes the attention analysis easier to interpret.

---

## 11. Open Questions Before Starting

- [ ] Are `gold_paragraphs` values always exact substrings of the cleaned verdict text, or might OCR cleaning have modified them? (Verify on 10 samples before committing to the positional pipeline.)
- [ ] For multi-`gold_paragraphs` records (2+ passages), how should we define "depth" — min, mean, or max position?
- [ ] GPU availability: does Qwen2.5-7B fit comfortably on the local GPU, or is a cloud instance (Colab A100, RunPod) required?
- [ ] Should the failure taxonomy include Phase I LC *successes* (faithfulness > 0.8) as a contrast class? This would turn it into a comparative analysis rather than a pure failure audit.
