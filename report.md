# When Context Is Not Enough: Long Context vs Retrieval-Augmented Generation on Indonesian Constitutional Court Verdicts

**Muhammad Iqbal Hilmy Izzulhaq**

*March 2026*

---

## Abstract

The recent expansion of large language model (LLM) context windows raises a practical question for document-grounded question answering: if an entire source document fits into the prompt, is retrieval-augmented generation (RAG) still necessary? This report studies that question on Indonesian Constitutional Court (*Mahkamah Konstitusi*, MK) verdicts using repository-grounded experimental artifacts from `results/`. The benchmark uses 50 stratified verdicts and 300 human-reviewed question-answer pairs spanning factual extractive, multi-section reasoning, structural, and boundary questions. Three architectures are compared: Long Context (LC), Simple RAG, and Advanced RAG.

Across both the Phase 1 baseline and the Phase 2 multi-model reruns, Simple RAG is the most reliable architecture. In Phase 1, Simple RAG achieves mean faithfulness of 0.845 on 300 queries, while the Long Context run records 0.451 on the 249 queries that returned scorable outputs; the remaining 51 LC records contain quota-exhaustion errors in the raw JSONL. In Phase 2, where Gemini Flash and GPT-4o Mini are each evaluated across all three architectures on complete 300-query runs, the ranking remains stable: Simple RAG > Advanced RAG > Long Context for both model families. The strongest complete same-model comparison is Gemini Flash, where Simple RAG reaches 0.840 faithfulness versus 0.616 for Long Context while costing $0.31 versus $7.77 over 300 queries.

The ablation study shows that "Advanced RAG" is not uniformly beneficial. Hybrid search is the single best component tested, reaching 0.907 mean faithfulness on the 100-question ablation subset. By contrast, query rewriting lowers faithfulness to 0.767 and cross-encoder reranking lowers it further to 0.725. Additional experiments reinforce the main result: on needle-in-a-haystack queries, Simple RAG reaches 0.906 faithfulness versus 0.683 for Long Context; in a small knowledge-update test on three unseen verdicts, Simple RAG attains perfect faithfulness (1.000), compared with 0.667 for Long Context and 0.333 for Advanced RAG. Taken together, the repository results support a clear conclusion: for factual legal QA over structured verdicts, targeted retrieval remains more faithful, cheaper, and more operationally robust than full-document prompt injection.

---

## 1. Introduction

Long-context LLMs are often presented as a potential replacement for retrieval pipelines. The underlying intuition is straightforward: if the model can read the whole document directly, then chunking, embedding, indexing, and ranking may appear unnecessary. For legal NLP, however, this intuition is not obviously correct. Legal documents are long, repetitive, highly structured, and precision-sensitive. A system can have access to the full text and still fail to focus on the correct passages.

Indonesian Constitutional Court verdicts are a particularly demanding test case. They are lengthy, formal, and sectioned by legal numbering conventions. Relevant facts may appear in procedural sections, argument sections, evidentiary sections, or the final ruling (*amar putusan*). Many questions are answerable from a small part of the document, but that part may be deeply buried in the full text. This makes the corpus suitable for testing whether retrieval helps concentrate signal more effectively than raw prompt expansion.

This report refines the project write-up into a publication-style narrative grounded in the repository artifacts available as of 27 March 2026. Methodological context is taken from `Runbook.md` and `Structure.md`, while all quantitative claims below are anchored to the committed experimental outputs summarized in Appendix A.

### 1.1 Research Questions

The study addresses four questions:

1. Does Long Context outperform RAG when an entire verdict can be placed in the prompt?
2. Does the architecture ranking hold across model families with different context-window characteristics?
3. Which Advanced RAG components help on this legal corpus, and which harm performance?
4. How do the architectures behave under adversarial retrieval depth and under corpus updates?

### 1.2 Main Contributions

1. A benchmark centered on 50 Indonesian Constitutional Court verdicts and 300 reviewed QA pairs.
2. A controlled empirical comparison of Long Context, Simple RAG, and Advanced RAG across Gemini Flash and GPT-4o Mini.
3. A component-level ablation showing that hybrid search helps, while query rewriting and reranking degrade performance on this corpus.
4. An evidence-based cost and latency analysis using the exact run artifacts committed under `results/`.

---

## 2. Corpus, Dataset, and Systems

### 2.1 Corpus Construction

According to the project runbook, the corpus pipeline starts from 887 raw MK verdict artifacts and proceeds through audit, cleaning, section extraction, metadata enrichment, and sampling. The final benchmark sample contains 50 verdicts stratified by document length:

- 15 short verdicts
- 20 medium verdicts
- 15 long verdicts

This stratified design matters because the central question is not only whether retrieval helps in aggregate, but whether it remains useful as document length grows.

### 2.2 Question-Answer Dataset

The repository's full QA dataset (`data/qa_dataset/qa_pairs_full.jsonl`) contains exactly 300 items across 50 verdicts. The observed question-type distribution is:

| Question type | Count |
|:---|---:|
| Factual extractive | 100 |
| Multi-section reasoning | 100 |
| Structural | 50 |
| Boundary | 50 |

This distribution gives the benchmark a mixed profile. It is not a pure extraction task, but it is also not dominated by open-ended legal reasoning. That balance is useful because it exposes the difference between retrieving the right evidence and merely generating plausible legal prose.

### 2.3 Architectures Evaluated

Three system families are evaluated throughout the repository:

**Long Context (LC).** The verdict text is injected directly into the prompt. No retrieval is used.

**Simple RAG.** Verdicts are chunked, embedded, and retrieved using dense retrieval; the retrieved chunks are placed in the prompt.

**Advanced RAG.** The pipeline augments Simple RAG with query rewriting, metadata filtering, hybrid search, and reranking. The ablation study tests these components incrementally.

### 2.4 Evaluation Metrics

The run artifacts expose the following metrics at the per-query level:

- `faithfulness`
- `hallucination_rate`
- `bertscore_f1`
- `latency_s`
- token counts and generation cost fields
- retrieval metrics such as `context_precision` and `context_recall` where applicable

Faithfulness is the primary metric in this report because the benchmark targets document-grounded legal QA. BERTScore remains useful as a semantic-similarity signal, but the results show that it can diverge from factual grounding. In this repository, BERTScore F1 is computed with `indolem/indobert-base-uncased` (IndoBERT) as the primary evaluation model, with multilingual BERT available only as a fallback if the primary model is unavailable.

---

## 3. Experimental Evidence from `results/`

### 3.1 Phase 1 Baseline

The Phase 1 baseline compares all three architectures using Gemini Flash over the 300-question benchmark. Table 1 reports the aggregate values from the committed Phase 1 outputs.

**Table 1. Phase 1 baseline from `results/phase1/*`.**

| Architecture | Scorable queries | Mean faithfulness | Mean BERTScore F1 | Mean hallucination rate | Zero-faithfulness cases | Mean input tokens | Total cost (USD) | Total latency |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Simple RAG | 300 | **0.845** | **0.510** | **0.155** | **25** | 1,990 | **0.3138** | 17.9 min |
| Advanced RAG | 300 | 0.764 | 0.490 | 0.236 | 47 | 1,339 | 0.2153 | 30.0 min |
| Long Context | 249 | 0.451 | 0.459 | 0.549 | 89 | 29,789 | 4.4824 | 14.7 min |

The Phase 1 result is directionally strong: Simple RAG outperforms both alternatives, and Long Context performs worst by a large margin. However, the LC run also needs to be interpreted carefully. The archived Phase 1 LC output contains 300 records, but 51 of them are error records without answers or faithfulness scores. These are not payload-format failures; the raw error fields show Gemini `429 RESOURCE_EXHAUSTED` quota-exhaustion errors tied to input-token limits. As a result, the Long Context mean in Table 1 is computed over 249 scorable outputs rather than the full benchmark.

Even with that caveat, the signal is clear. On completed queries, Simple RAG exceeds Long Context by 0.394 absolute faithfulness points. The cost difference is also substantial: Simple RAG processes the 300-query benchmark for roughly $0.31, whereas the incomplete Long Context run already consumes $4.48.

### 3.2 Phase 2 Multi-Model Comparison

Phase 2 provides the cleaner architectural comparison because all six cells are complete 300-query runs. Table 2 therefore serves as the strongest basis for cross-architecture comparison.

**Table 2. Phase 2 full factorial results with bootstrap 95% confidence intervals for mean faithfulness.**

| Model | Architecture | Queries | Mean faithfulness | 95% CI | Mean BERTScore F1 | Mean hallucination rate | Zero-faithfulness cases |
|:---|:---|---:|---:|:---:|---:|---:|---:|
| Gemini Flash | Simple RAG | 300 | **0.840** | [0.803, 0.877] | 0.507 | **0.160** | **32** |
| Gemini Flash | Advanced RAG | 300 | 0.762 | [0.718, 0.805] | 0.489 | 0.238 | 50 |
| Gemini Flash | Long Context | 300 | 0.616 | [0.565, 0.666] | **0.548** | 0.384 | 85 |
| GPT-4o Mini | Simple RAG | 300 | **0.857** | [0.816, 0.893] | 0.681 | **0.143** | **39** |
| GPT-4o Mini | Advanced RAG | 300 | 0.780 | [0.733, 0.825] | 0.647 | 0.220 | 61 |
| GPT-4o Mini | Long Context | 300 | 0.524 | [0.469, 0.578] | **0.728** | 0.476 | 130 |

Two observations matter most.

First, the architecture ordering is stable across both models: Simple RAG performs best, Advanced RAG is second, and Long Context is worst. This matters more than the exact means because it shows that the result is not a one-off artifact of a single run configuration.

Second, Long Context again posts the highest BERTScore while remaining the least faithful. That pattern is visible in both model families. For GPT-4o Mini, Long Context reaches the highest BERTScore (0.728) but the lowest faithfulness (0.524). For Gemini Flash, Long Context similarly leads on BERTScore (0.548) but trails on faithfulness (0.616). In other words, on this benchmark, semantic-overlap metrics alone would overstate LC quality.

Third, the main Phase 2 comparisons are statistically well supported. Paired one-sided Wilcoxon signed-rank tests on per-question faithfulness scores show that Simple RAG significantly outperforms Advanced RAG for both Gemini Flash (`p = 0.00154`) and GPT-4o Mini (`p = 0.00176`), and it significantly outperforms Long Context for both Gemini Flash (`p = 3.09e-12`) and GPT-4o Mini (`p = 2.43e-17`). These tests do not replace a fuller inferential treatment, but they materially strengthen the central claim.

### 3.3 Cost and Latency

The repository artifacts show that the most faithful systems are also the most cost-efficient. Table 3 consolidates the Phase 2 cost evidence.

**Table 3. Phase 2 efficiency summary.**

| Model | Architecture | Mean input tokens | Mean cost/query (USD) | Total cost over 300 queries (USD) | Total latency |
|:---|:---|---:|---:|---:|---:|
| Gemini Flash | Simple RAG | 1,990 | 0.001044 | 0.3131 | 18.1 min |
| Gemini Flash | Advanced RAG | 1,324 | 0.000710 | 0.2130 | 31.5 min |
| Gemini Flash | Long Context | 51,719 | 0.025913 | 7.7740 | 18.2 min |
| GPT-4o Mini | Simple RAG | 2,332 | 0.000210 | 0.0630 | 23.6 min |
| GPT-4o Mini | Advanced RAG | 1,550 | **0.000146** | **0.0438** | 38.3 min |
| GPT-4o Mini | Long Context | 45,051 | 0.003421 | 1.0262 | 69.6 min |

For same-model comparisons, the trade-off is straightforward. Under Gemini Flash, Simple RAG is about 24.8x cheaper than Long Context while being substantially more faithful (0.840 vs 0.616). Under GPT-4o Mini, Simple RAG is about 16.3x cheaper than Long Context and again more faithful (0.857 vs 0.524).

Across all six Phase 2 cells, the cheapest condition is GPT-4o Mini + Advanced RAG at $0.0438 total cost, and the most expensive is Gemini Flash + Long Context at $7.7740. That is an approximately 177x cost spread across tested configurations. Importantly, the expensive configuration is not the best-performing one.

### 3.4 Ablation Study

The ablation artifacts are especially informative because they show why the full Advanced RAG pipeline underperforms Simple RAG. Table 4 reports the 100-question subset results.

**Table 4. Ablation study from `results/ablation/`.**

| Condition | Mean faithfulness | Mean BERTScore F1 | Mean hallucination rate | Mean latency/query | Mean cost/query (USD) |
|:---|---:|---:|---:|---:|---:|
| Baseline Simple RAG | 0.815 | 0.513 | 0.185 | 3.81 s | 0.001037 |
| + Query Rewrite | 0.767 | 0.482 | 0.233 | 5.76 s | 0.000964 |
| + Metadata Filter | 0.840 | 0.507 | 0.160 | 3.82 s | 0.001035 |
| + Hybrid Search | **0.907** | **0.533** | **0.093** | **3.75 s** | 0.001174 |
| + Reranking | 0.725 | 0.497 | 0.275 | 4.18 s | **0.000700** |
| Full Advanced RAG | 0.803 | 0.526 | 0.197 | 6.26 s | 0.000710 |

This table supports three concrete claims.

First, hybrid search is highly beneficial on this legal corpus. It is the single best ablation condition on faithfulness and hallucination rate, and it does so without increasing latency relative to baseline.

Second, query rewriting is harmful in its current form. It reduces faithfulness from 0.815 to 0.767 while also increasing latency.

Third, reranking is the most damaging step tested. Once reranking is added, mean faithfulness drops to 0.725, the lowest score in the ablation sequence. A plausible explanation is model mismatch: the reranker used in the pipeline is a general-domain `ms-marco` MiniLM cross-encoder trained primarily on English retrieval signals rather than Bahasa Indonesia legal prose. In this setting, it likely overweights superficial query-passage similarity and underweights domain-specific legal cues such as article references, formal court terminology, and section-local procedural language. Because reranking also prunes the candidate set aggressively, a modest ranking error can become an unrecoverable evidence loss.

### 3.5 Needle-in-a-Haystack Evaluation

The needle-in-a-haystack experiment evaluates queries whose gold evidence appears deep in the verdict text. Table 5 reports the committed summary.

**Table 5. Needle-in-a-haystack results.**

| Architecture | Queries | NIAH accuracy | Mean faithfulness | Mean BERTScore F1 | Zero-faithfulness cases | Total latency |
|:---|---:|---:|---:|---:|---:|---:|
| Simple RAG | 30 | **0.8333** | **0.9056** | 0.5058 | **1** | 1.80 min |
| Advanced RAG | 30 | 0.7667 | 0.8167 | 0.5075 | 4 | 3.19 min |
| Long Context | 30 | 0.5000 | 0.6833 | **0.5476** | 4 | **1.68 min** |

The NIAH results reinforce the main finding rather than weakening it. Simple RAG is markedly better at locating and grounding deep evidence. Long Context remains competitive only on superficial similarity metrics and raw runtime, not on factual success.

### 3.6 Knowledge Update Scenario

The repository also includes a small update experiment in which three unseen verdicts are added and then queried.

**Table 6. Knowledge-update results.**

| Architecture | Queries | Mean faithfulness | Mean BERTScore F1 | Mean latency/query | Total cost (USD) |
|:---|---:|---:|---:|---:|---:|
| Simple RAG | 3 | **1.000** | 0.2675 | **3.25 s** | **0.002863** |
| Advanced RAG | 3 | 0.333 | **0.3052** | 8.44 s | 0.002198 |
| Long Context | 3 | 0.667 | 0.2708 | 5.83 s | 0.103845 |

The recorded update latencies show that indexing the three new verdicts takes about 0.686 s, 4.153 s, and 18.623 s respectively. Long Context has zero indexing overhead by design, but the absence of indexing does not translate into the best QA performance. On this small test, Simple RAG remains the most faithful and the fastest at inference time.

The sample size here is too small for strong generalization, but the direction is consistent with the rest of the benchmark.

---

## 4. Discussion

### 4.1 Why Simple RAG Wins on This Corpus

The most plausible explanation is not that Long Context cannot "see" the answer, but that it does not reliably prioritize the answer-bearing passages. MK verdicts are long, formulaic, and internally structured. For many benchmark questions, only a few short passages are needed. Simple RAG reduces the search space before generation. That lets the model reason over a concentrated evidence set instead of a full-document prompt dominated by procedural boilerplate and unrelated legal discussion.

This interpretation is consistent with three empirical patterns in the artifacts:

1. Simple RAG beats Long Context on both Gemini Flash and GPT-4o Mini.
2. Hybrid search further improves faithfulness, suggesting lexical cues remain important.
3. Needle-in-a-haystack performance strongly favors retrieval, which is exactly where prompt-wide attention should struggle most.

### 4.2 Why "More RAG" Does Not Automatically Help

The ablation results show that additional retrieval components introduce failure modes of their own. The current query-rewrite step likely broadens or distorts the original information need. The reranker appears especially brittle, possibly because it is not adapted to Indonesian legal language and because it prunes aggressively enough to discard relevant chunks. Metadata filtering is mildly helpful, but only hybrid search produces a clear and substantial gain.

This matters for system design. The correct lesson is not "Advanced RAG is bad." The correct lesson is that retrieval pipelines must be validated component by component. A compact, well-calibrated retrieval stack can outperform both brute-force LC and a more elaborate but mismatched retrieval cascade.

### 4.3 The BERTScore Paradox

One of the clearest measurement lessons in the repository is that BERTScore can rank Long Context highly even when faithfulness is low. This likely occurs because LC answers are often longer and semantically closer at the surface level to reference answers, while still including unsupported content. In legal QA, such answers are risky because a partially correct but embellished response can still be unusable.

For publication purposes, this means faithfulness should remain the lead metric, and semantic-overlap scores should be presented as complementary rather than decisive.

### 4.4 Operational Reliability

The Phase 1 Long Context run exposes a practical deployment concern: very large prompts are not only expensive, but also more vulnerable to quota and throughput constraints. The 51 unscored LC records in Phase 1 are not a theoretical issue; they are documented in the repository artifacts as quota-exhaustion failures. Even when Long Context is later rerun successfully in Phase 2, the Phase 1 instability remains relevant because production systems must handle repeated workloads, not just ideal reruns.

---

## 5. Limitations

Several limitations should be stated explicitly.

1. The benchmark is domain-specific. The results are grounded in Indonesian Constitutional Court verdicts and may not transfer directly to other legal corpora or other languages.
2. The knowledge-update experiment is very small (`n = 3`) and should be interpreted as directional.
3. The current report is grounded in committed result artifacts, not in the full planned notebook suite described in `Runbook.md`. Basic confidence intervals and paired non-parametric significance tests are now reported for the main Phase 2 faithfulness comparisons, but fuller inferential treatment, including multiple-testing strategy and human legal spot-check summaries, would still improve publication readiness.
4. The report relies on repository documentation for parts of the pipeline description. The quantitative claims, however, are restricted to what can be verified from the result files themselves.

---

## 6. Conclusion

The repository evidence supports a clear conclusion: on Indonesian Constitutional Court verdicts, expanding the context window does not eliminate the need for retrieval. Across the complete Phase 2 runs, Simple RAG is the best architecture on faithfulness for both Gemini Flash and GPT-4o Mini. Across the ablation study, hybrid search is the strongest individual enhancement, while query rewriting and reranking reduce quality. Across the additional experiments, retrieval remains strongest under deep-evidence queries and under small corpus updates.

For factual legal QA, the practical recommendation is therefore not to choose between retrieval and context capacity in the abstract. It is to use context capacity selectively, after retrieval has already concentrated the relevant evidence. In this project, that design consistently yields the best balance of faithfulness, cost, and operational robustness.

---

## Appendix A. Artifact Grounding

The main quantitative claims in this report are grounded in the following repository artifacts:

- `results/phase1/simple_rag/run_20260320_080919_bs.jsonl`
- `results/phase1/advanced_rag/run_20260320_164547_bs.jsonl`
- `results/phase1/lc/run_20260320_073844_bs.jsonl`
- `results/phase2/gemini_flash/*/results_clean_bs.jsonl`
- `results/phase2/gpt4o/*/results_clean_bs.jsonl`
- `results/ablation/*/results_clean_bs.jsonl`
- `results/additional/niah/niah_summary.csv`
- `results/additional/knowledge_update/update_latency.csv`
- `report_totals.txt`

---

## References

- Es, S., James, J., Espinosa-Anke, L., and Schockaert, S. (2024). RAGAS: Automated Evaluation of Retrieval Augmented Generation. arXiv:2309.15217.
- Guha, N., et al. (2023). LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models. NeurIPS.
- Hendrycks, D., et al. (2021). CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review. NeurIPS.
- Koto, F., Rahimi, A., Lau, J. H., and Baldwin, T. (2020). IndoLEM and IndoBERT: A Benchmark Dataset and Pre-trained Language Model for Indonesian NLP. COLING.
- Li, N., et al. (2024). Long-Context LLMs Struggle with Long In-Context Learning. arXiv:2404.02060.
- Liu, N. F., et al. (2023). Lost in the Middle: How Language Models Use Long Contexts. TACL.
- Xu, F., et al. (2024). Retrieval Meets Long Context Large Language Models. ICML.
