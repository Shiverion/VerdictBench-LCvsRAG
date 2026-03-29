# When Context Is Not Enough: Long Context vs Retrieval-Augmented Generation on Indonesian Constitutional Court Verdicts

**Muhammad Iqbal Hilmy Izzulhaq**

*March 2026*

---

## Abstract

The recent expansion of large language model (LLM) context windows raises a practical question for document-grounded question answering: if an entire source document fits into the prompt, is retrieval-augmented generation (RAG) still necessary? This report evaluates that question on 50 Indonesian Constitutional Court (*Mahkamah Konstitusi*, MK) verdicts and 300 human-reviewed question-answer pairs.

Simple RAG is the most reliable architecture in every complete comparison. In Phase 1, it reaches mean faithfulness 0.845 over 300 queries, whereas Long Context (LC) reaches 0.451 over 249 scorable queries; the remaining 51 LC failures are quota-exhaustion errors, all concentrated in the long-verdict stratum. In Phase 2, the ranking remains stable for both Gemini Flash and GPT-4o Mini: Simple RAG > Advanced RAG > Long Context. Under Gemini Flash, Simple RAG attains 0.840 faithfulness versus 0.616 for LC while costing $0.31 versus $7.77 over 300 queries.

Effect sizes support the same conclusion. For the main Phase 2 comparisons, Cohen's *d* ranges from 0.206 to 0.220 for Simple RAG versus Advanced RAG and from 0.582 to 0.803 for Simple RAG versus Long Context. The ablation study shows that hybrid search is the most beneficial component (0.907 faithfulness), whereas query rewriting and cross-encoder reranking reduce quality. On this legal QA benchmark, targeted retrieval remains more faithful, cheaper, and operationally more robust than full-document prompt injection.

---

## 1. Introduction

Long-context LLMs are often presented as a potential replacement for retrieval pipelines. The intuition is straightforward: if the model can read the whole document directly, chunking, embedding, indexing, and ranking may appear unnecessary. For legal NLP, however, this intuition is not obviously correct. Legal documents are long, repetitive, highly structured, and precision-sensitive. A system can have access to the full text and still fail to focus on the correct passages.

Indonesian Constitutional Court verdicts are a demanding test case. Relevant facts may appear in procedural sections, argument sections, evidentiary sections, or the final ruling (*amar putusan*). Many benchmark questions are answerable from a small part of the document, but that part may be deeply buried in the full text. This makes the corpus well suited to testing whether retrieval helps concentrate signal more effectively than raw prompt expansion.

This report refines the project write-up into a publication-style narrative grounded in the repository artifacts available as of 29 March 2026. Methodological context is taken from `Runbook.md` and `Structure.md`, while all quantitative claims below are anchored to the committed experimental outputs summarized in Appendix A.

### 1.1 Research Questions

The study addresses four questions:

1. Does Long Context outperform RAG when an entire verdict can be placed in the prompt?
2. Does the architecture ranking hold across model families with different context-window characteristics?
3. Which Advanced RAG components help on this legal corpus, and which harm performance?
4. How do the architectures behave under adversarial retrieval depth?

### 1.2 Main Contributions

1. A benchmark centered on 50 Indonesian Constitutional Court verdicts and 300 reviewed QA pairs.
2. A controlled empirical comparison of Long Context, Simple RAG, and Advanced RAG across Gemini Flash and GPT-4o Mini.
3. A component-level ablation showing that hybrid search helps, while query rewriting and reranking degrade performance on this corpus.
4. An evidence-based cost, latency, and operational-reliability analysis using the exact run artifacts committed under `results/`.

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

### 2.3 Inter-Annotator Agreement

To assess annotation consistency, a second annotator independently reviewed a randomly drawn 10% overlap set of 30 QA pairs, blind to the primary annotator's labels. We report observed agreement, Cohen's kappa, and the label distribution together because kappa alone would be misleading on this task.

**Table A. Inter-annotator agreement on the 30-item overlap set.**

| Metric | Value |
|:---|---:|
| Shared items | 30 |
| Observed agreement | 0.900 |
| Cohen's kappa | 0.000 |
| Annotator 1: `accepted` | 27 |
| Annotator 1: `modified` | 3 |
| Annotator 2: `accepted` | 30 |

The gap between 0.900 observed agreement and kappa = 0.000 is not a contradiction. It is a textbook instance of the prevalence-kappa paradox documented by Feinstein and Cicchetti (1990). When one label dominates, the chance-agreement baseline approaches the observed-agreement ceiling and kappa collapses toward zero even when annotators are substantively consistent. In this overlap set, both annotators agreed on 27 of 30 items outright, and the 3 disagreements reflect minor phrasing judgments rather than factual disputes. The annotation task itself limits subjectivity because gold answers must be verifiable from explicit source text.

We therefore treat 0.900 observed agreement as the primary reliability signal and report kappa alongside it as a transparency measure rather than a standalone quality indicator.

### 2.4 Architectures Evaluated

Three system families are evaluated throughout the repository:

**Long Context (LC).** The verdict text is injected directly into the prompt. No retrieval is used.

**Simple RAG.** Verdicts are chunked, embedded, and retrieved using dense retrieval (Johnson et al., 2021); the retrieved chunks are placed in the prompt.

**Advanced RAG.** The pipeline augments Simple RAG with query rewriting, metadata filtering, hybrid BM25+dense search with Reciprocal Rank Fusion (Cormack et al., 2009), and cross-encoder reranking (Nogueira and Cho, 2019; Reimers and Gurevych, 2019). The ablation study tests these components incrementally.

### 2.5 Evaluation Metrics

The run artifacts expose the following metrics at the per-query level:

- `faithfulness`
- `hallucination_rate`
- `bertscore_f1`
- `latency_s`
- token counts and generation cost fields
- retrieval metrics such as `context_precision` and `context_recall` where applicable

Faithfulness is the primary metric in this report because the benchmark targets document-grounded legal QA. BERTScore (Zhang et al., 2020) remains useful as a semantic-similarity signal, but the results show that it can diverge from factual grounding. In this repository, BERTScore F1 is computed with `indolem/indobert-base-uncased` (IndoBERT) as the primary evaluation model (Koto et al., 2020), with multilingual BERT available only as a fallback if the primary model is unavailable. Coverage is complete for every scorable record reported in Tables 1 to 4: all Phase 1 scored outputs, all six Phase 2 cells, and all ablation conditions have populated `bertscore_f1` values.

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

The Phase 1 result is directionally strong: Simple RAG outperforms both alternatives, and Long Context performs worst by a large margin. However, the LC run must be interpreted carefully. The archived Phase 1 LC output contains 300 records, but 51 of them are error records without answers or faithfulness scores. These are not payload-format failures; the raw error fields show Gemini `429 RESOURCE_EXHAUSTED` quota-exhaustion errors tied to input-token limits. As a result, the Long Context mean in Table 1 is computed over 249 scorable outputs rather than the full benchmark.

The failure pattern is also highly structured rather than random. Joining the 51 unscored LC records to the benchmark strata in `data/metadata/sample_50.csv` shows that all 51 occur in the `long` stratum. That corresponds to a 56.7% failure rate for long-verdict queries (51/90) and a 0% failure rate for both short and medium verdicts (0/210 combined). This makes the Phase 1 LC breakdown substantively informative: the architecture degrades exactly where long-context prompting is supposed to be most advantageous.

Even with that caveat, the signal is clear. On completed queries, Simple RAG exceeds Long Context by 0.394 absolute faithfulness points. Restricting the comparison to the 249 questions with scorable outputs in both systems, Simple RAG also significantly outperforms Long Context under a paired one-sided Wilcoxon signed-rank test (`p = 3.29e-21`). The cost difference is also substantial: Simple RAG processes the 300-query benchmark for roughly $0.31, whereas the incomplete Long Context run already consumes $4.48.

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

**Table 2A. Phase 2 paired effect sizes on per-question faithfulness.**

| Model | Comparison | Mean difference | Cohen's *d* | 95% bootstrap CI |
|:---|:---|---:|---:|:---:|
| Gemini Flash | Simple RAG vs Advanced RAG | 0.078 | 0.220 | [0.065, 0.379] |
| Gemini Flash | Simple RAG vs Long Context | 0.224 | 0.582 | [0.411, 0.753] |
| Gemini Flash | Advanced RAG vs Long Context | 0.146 | 0.356 | [0.193, 0.525] |
| GPT-4o Mini | Simple RAG vs Advanced RAG | 0.077 | 0.206 | [0.049, 0.361] |
| GPT-4o Mini | Simple RAG vs Long Context | 0.333 | 0.803 | [0.629, 0.988] |
| GPT-4o Mini | Advanced RAG vs Long Context | 0.256 | 0.578 | [0.417, 0.750] |

Three observations matter most.

First, the architecture ordering is stable across both models: Simple RAG performs best, Advanced RAG is second, and Long Context is worst. This matters more than the exact means because it shows that the result is not a one-off artifact of a single run configuration.

Second, Long Context again posts the highest BERTScore while remaining the least faithful. For GPT-4o Mini, Long Context reaches the highest BERTScore (0.728) but the lowest faithfulness (0.524). For Gemini Flash, Long Context similarly leads on BERTScore (0.548) but trails on faithfulness (0.616). On this benchmark, semantic-overlap metrics alone would therefore overstate LC quality.

Third, the main Phase 2 comparisons are statistically and practically well supported. Paired one-sided Wilcoxon signed-rank tests on per-question faithfulness scores show that Simple RAG significantly outperforms Advanced RAG for both Gemini Flash (`p = 0.00154`) and GPT-4o Mini (`p = 0.00176`), and it significantly outperforms Long Context for both Gemini Flash (`p = 3.09e-12`) and GPT-4o Mini (`p = 2.43e-17`). Table 2A shows the corresponding effect sizes. The Simple-RAG-versus-Advanced-RAG gains are small but consistent (`d = 0.220` for Gemini Flash; `d = 0.206` for GPT-4o Mini), whereas the Simple-RAG-versus-Long-Context gains are medium to large (`d = 0.582` and `d = 0.803`). The central ranking is therefore supported not only by significance tests but also by practically meaningful separation, especially against Long Context.

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

First, hybrid BM25+dense search (Cormack et al., 2009) is highly beneficial on this legal corpus. It is the single best ablation condition on faithfulness and hallucination rate, and it does so without increasing latency relative to baseline.

Second, query rewriting is harmful in its current form. It reduces faithfulness from 0.815 to 0.767 while also increasing latency.

Third, reranking is the most damaging step tested. Once reranking is added, mean faithfulness drops to 0.725, the lowest score in the ablation sequence. A plausible explanation is model mismatch: the reranker used in the pipeline is a general-domain `ms-marco` MiniLM cross-encoder (Nogueira and Cho, 2019; Reimers and Gurevych, 2019) trained primarily on English retrieval signals rather than Bahasa Indonesia legal prose. In this setting, it likely overweights superficial query-passage similarity and underweights domain-specific legal cues such as article references, formal court terminology, and section-local procedural language. Because reranking also prunes the candidate set aggressively, a modest ranking error can become an unrecoverable evidence loss.

### 3.5 Needle-in-a-Haystack Evaluation

The needle-in-a-haystack experiment evaluates queries whose gold evidence appears deep in the verdict text. Table 5 reports the committed summary.

**Table 5. Needle-in-a-haystack results.**

| Architecture | Queries | NIAH accuracy | Mean faithfulness | Mean BERTScore F1 | Zero-faithfulness cases | Total latency |
|:---|---:|---:|---:|---:|---:|---:|
| Simple RAG | 30 | **0.8333** | **0.9056** | 0.5058 | **1** | 1.80 min |
| Advanced RAG | 30 | 0.7667 | 0.8167 | 0.5075 | 4 | 3.19 min |
| Long Context | 30 | 0.5000 | 0.6833 | **0.5476** | 4 | **1.68 min** |

The NIAH results reinforce the main finding rather than weakening it. Simple RAG is markedly better at locating and grounding deep evidence. Long Context remains competitive only on superficial similarity metrics and raw runtime, not on factual success.

---

## 4. Discussion

### 4.1 Why Simple RAG Wins on This Corpus

The most plausible explanation is not that Long Context cannot "see" the answer, but that it does not reliably prioritize the answer-bearing passages. MK verdicts are long, formulaic, and internally structured. For many benchmark questions, only a few short passages are needed. Simple RAG reduces the search space before generation. That lets the model reason over a concentrated evidence set instead of a full-document prompt dominated by procedural boilerplate and unrelated legal discussion.

This interpretation is consistent with three empirical patterns in the artifacts:

1. Simple RAG beats Long Context on both Gemini Flash and GPT-4o Mini.
2. Hybrid search further improves faithfulness, suggesting lexical cues remain important (Cormack et al., 2009).
3. Needle-in-a-haystack performance strongly favors retrieval, which is exactly where prompt-wide attention should struggle most (Liu et al., 2023; Li et al., 2024).

### 4.2 Why "More RAG" Does Not Automatically Help

The ablation results show that additional retrieval components introduce failure modes of their own. The current query-rewrite step likely broadens or distorts the original information need. The reranker appears especially brittle, possibly because it is not adapted to Indonesian legal language and because it prunes aggressively enough to discard relevant chunks. Metadata filtering is mildly helpful, but only hybrid search produces a clear and substantial gain.

This matters for system design. The correct lesson is not "Advanced RAG is bad." The correct lesson is that retrieval pipelines must be validated component by component. A compact, well-calibrated retrieval stack can outperform both brute-force LC and a more elaborate but mismatched retrieval cascade.

### 4.3 The BERTScore Paradox

One of the clearest measurement lessons in the repository is that BERTScore (Zhang et al., 2020) can rank Long Context highly even when faithfulness is low. This likely occurs because LC answers are often longer and semantically closer at the surface level to reference answers, while still including unsupported content. In legal QA, such answers are risky because a partially correct but embellished response can still be unusable.

For publication purposes, this means faithfulness should remain the lead metric, and semantic-overlap scores should be presented as complementary rather than decisive.

### 4.4 Operational Reliability

The Phase 1 Long Context run exposes a practical deployment concern: very large prompts are not only expensive, but also more vulnerable to quota and throughput constraints. The 51 unscored LC records in Phase 1 are not a theoretical issue; they are documented in the repository artifacts as quota-exhaustion failures. Even when Long Context is later rerun successfully in Phase 2, the Phase 1 instability remains relevant because production systems must handle repeated workloads, not just ideal reruns.

---

## 5. Limitations

Several limitations should be stated explicitly.

1. **Domain specificity.** The results are grounded in Indonesian Constitutional Court verdicts and may not transfer directly to other legal corpora or other languages.
2. **Knowledge-update evidence.** The repository contains a three-item knowledge-update pilot, but `n = 3` is too small to support a paper-level claim. For the present submission, that pilot is treated as exploratory and excluded from the main empirical argument. A defensible follow-up would require at least `n >= 15` unseen verdicts.
3. **Inferential scope.** The main Phase 2 comparisons now report bootstrap confidence intervals, paired Wilcoxon tests, and Cohen's *d*, but no family-wise or false-discovery correction is applied across all exploratory pairwise comparisons in the repository.
4. **Human legal accuracy spot-check.** A 10% human legal accuracy review (0/1/2 scale) across all Phase 1 conditions has not yet been completed and will be included before submission.
5. **Inter-annotator agreement and the prevalence-kappa paradox.** The 30-item overlap set yields 0.900 observed agreement alongside kappa = 0.000. As discussed in Section 2.3, this gap is fully explained by the near-uniform `accepted` label distribution across both annotators (Feinstein and Cicchetti, 1990), not by genuine rater disagreement. We report both figures transparently so readers can assess the annotation quality directly, rather than relying on a single summary statistic that would misrepresent the underlying consistency.

---

## 6. Conclusion

The repository evidence supports a clear conclusion: on Indonesian Constitutional Court verdicts, expanding the context window does not eliminate the need for retrieval. Across the complete Phase 2 runs, Simple RAG is the best architecture on faithfulness for both Gemini Flash and GPT-4o Mini. Across the ablation study, hybrid search is the strongest individual enhancement, while query rewriting and reranking reduce quality. Under deep-evidence queries, retrieval remains strongest where full-document prompting should have been most competitive.

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
- `data/metadata/sample_50.csv`
- `data/qa_dataset/iaa_existing_overlap/kappa_summary.json`
- `report_totals.txt`

---

## References

- Cormack, G. V., Clarke, C. L. A., and Buettcher, S. (2009). Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. *SIGIR 2009*, 758-759.
- Es, S., James, J., Espinosa-Anke, L., and Schockaert, S. (2024). RAGAS: Automated Evaluation of Retrieval Augmented Generation. *arXiv:2309.15217*.
- Feinstein, A. R. and Cicchetti, D. V. (1990). High agreement but low kappa: I. The problems of two paradoxes. *Journal of Clinical Epidemiology*, 43(6), 543-549.
- Guha, N., et al. (2023). LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models. *NeurIPS*.
- Guu, K., Lee, K., Tung, Z., Pasupat, P., and Chang, M. (2020). REALM: Retrieval-Augmented Language Model Pre-Training. *ICML*.
- Hendrycks, D., et al. (2021). CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review. *NeurIPS*.
- Johnson, J., Douze, M., and Jegou, H. (2021). Billion-Scale Similarity Search with GPUs. *IEEE Transactions on Big Data*, 7(3), 535-547.
- Koto, F., Rahimi, A., Lau, J. H., and Baldwin, T. (2020). IndoLEM and IndoBERT: A Benchmark Dataset and Pre-trained Language Model for Indonesian NLP. *COLING*.
- Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.
- Li, T., Zhang, G., Do, Q. D., Yue, X., and Chen, W. (2024). Long-context LLMs Struggle with Long In-context Learning. *Transactions on Machine Learning Research (TMLR)*. arXiv:2404.02060.
- Liu, N. F., et al. (2023). Lost in the Middle: How Language Models Use Long Contexts. *Transactions of the Association for Computational Linguistics (TACL)*, 12, 157-173.
- Nogueira, R. and Cho, K. (2019). Passage Re-ranking with BERT. *arXiv:1901.04085*.
- Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP*. arXiv:1908.10084.
- Robertson, S. and Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*.
- Xu, P., Ping, W., Wu, X., McAfee, L., Zhu, C., Liu, Z., Subramanian, S., Bakhturina, E., Shoeybi, M., and Catanzaro, B. (2024). Retrieval Meets Long Context Large Language Models. *ICLR 2024*. arXiv:2310.03025.
- Zhang, T., et al. (2020). BERTScore: Evaluating Text Generation with BERT. *ICLR*.
