# When Context Is Not Enough: Benchmarking Long Context vs Retrieval-Augmented Generation on Indonesian Constitutional Court Verdicts

**Muhammad Iqbal Hilmy Izzulhaq**

*Independent Researcher*
*AI Engineering Consultant & Trainer, RevoU Jakarta*
*github.com/Shiverion/VerdictBench-LCvsRAG*

---

## Abstract

The recent expansion of large language model (LLM) context windows raises a practical question for document-grounded question answering: if an entire source document fits into the prompt, is retrieval-augmented generation (RAG) still necessary? We evaluate this question on 50 Indonesian Constitutional Court (*Mahkamah Konstitusi*, MK) verdicts and 300 human-reviewed question-answer pairs spanning four cognitive types.

Three architectures are compared across two model families: Long Context (LC), Simple RAG, and Advanced RAG, using Gemini 2.5 Flash and GPT-4o Mini. An evaluation bug in the original analysis suppressed LC faithfulness scores by judging answers against a 503-character document preview rather than the full generation context; all faithfulness figures reported here reflect a corrected oracle re-evaluation in which all three architectures are judged against human-verified gold evidence paragraphs (see Section 3.5). Under this corrected metric, Long Context achieves faithfulness statistically indistinguishable from Simple RAG in Phase 2 (Gemini Flash: d = 0.008, p = 0.43; GPT-4o Mini: d = 0.017, p = 0.49); both significantly outperform Advanced RAG (d = 0.175–0.202). A component-level ablation shows that Baseline Simple RAG achieves the highest gold-evidence faithfulness (0.604); every augmentation component reduces it, with hybrid search least harmful (0.585, −1.9 pp) and query rewriting most harmful (0.453, −15.1 pp). Length-sensitivity analysis reveals that LC faithfulness quality *increases* with verdict length (0.578/0.606/0.628 short/medium/long among answered queries); the poor overall long-verdict performance (coverage-adjusted 0.272) reflects a 56.7% operational non-response rate, not faithfulness quality degradation. On this legal QA benchmark, targeted retrieval achieves comparable faithfulness to full-document prompt injection at 25× lower cost with substantially better operational reliability on long documents.

---

## 1. Introduction

Long-context LLMs are frequently presented as a potential replacement for retrieval pipelines. The intuition is straightforward: if the model can ingest an entire document in a single prompt pass, the engineering overhead of chunking, embedding, indexing, and retrieval appears superfluous. For legal NLP, however, this intuition is not obviously correct. Legal documents are long, repetitive, highly structured, and precision-sensitive. A system can have access to the full text and still fail to locate and faithfully ground specific facts.

Indonesian Constitutional Court verdicts are a particularly demanding test case. These documents are lengthy (15,000–80,000 tokens), formally structured by legal numbering conventions, and rich in procedurally formulaic prose. Relevant facts may appear in procedural sections, argument sections, evidentiary records, or the final ruling (*amar putusan*). This structure makes MK verdicts well-suited to testing whether retrieval provides meaningful signal concentration beyond raw context expansion.

This paper makes four contributions:

1. The first systematic LC vs RAG comparison on Indonesian constitutional legal text, with a reusable benchmark of 50 verdicts and 300 human-reviewed QA pairs.
2. A controlled empirical comparison of Long Context, Simple RAG, and Advanced RAG across Gemini 2.5 Flash and GPT-4o Mini, with effect sizes and bootstrap confidence intervals.
3. A component-level ablation showing that, under oracle gold-evidence evaluation, Baseline Simple RAG is the strongest condition and every augmentation reduces faithfulness — with cross-encoder reranking most harmful due to language and domain mismatch.
4. A length-sensitivity and needle-in-a-haystack analysis showing that LC operational reliability collapses on the longest documents even though LC faithfulness quality is highest on those same verdicts, separating task-completion failure from output-quality failure.

---

## 2. Related Work

RAG was formalised by Lewis et al. (2020) as a method for combining parametric LLM knowledge with non-parametric document retrieval. Dense retrieval relies on approximate nearest-neighbour search over learned embeddings (Johnson et al., 2021), while hybrid architectures combine sparse BM25 (Robertson and Zaragoza, 2009) with dense retrieval via Reciprocal Rank Fusion (Cormack et al., 2009). Retrieval quality evaluation frameworks such as RAGAS (Es et al., 2024) and multi-hop reasoning approaches (Guu et al., 2020) have further developed the RAG ecosystem.

The proliferation of million-token context windows has prompted direct comparisons of LC versus RAG. Liu et al. (2023) demonstrate a "lost in the middle" effect in which LLMs systematically underweight evidence positioned in the interior of long contexts. Xu et al. (2024) show that retrieval-augmented LLMs outperform longer-context baselines on knowledge-intensive tasks at a fraction of the compute cost. Li et al. (2024) introduce LongICLBench and find that long-context LLMs struggle to exploit full context windows even when relevant information is present — a finding consistent with our NIAH results.

Cross-encoder reranking (Nogueira and Cho, 2019; Reimers and Gurevych, 2019) improves retrieval precision in English general-domain settings but carries domain-transfer risk when applied to low-resource languages. In the legal domain, LegalBench (Guha et al., 2023) and CUAD (Hendrycks et al., 2021) establish English benchmarks, but no comparable benchmark exists for Indonesian legal text. Indonesian NLP has received increasing attention following IndoBERT and IndoLEM (Koto et al., 2020), but legal-domain resources in Bahasa Indonesia remain absent. This work is the first systematic LC vs RAG comparison on Indonesian constitutional legal text.

---

## 3. Methodology

### 3.1 Corpus Construction

The corpus pipeline processes 887 raw MK verdict artifacts extracted from the official mkri.id portal through five sequential stages: audit, cleaning, section extraction, metadata enrichment, and stratified sampling. The final benchmark sample contains **50 verdicts** stratified by document length: 15 short, 20 medium, and 15 long. Cleaning normalises OCR artifacts including CRLF removal, soft-hyphen stripping, and Unicode standardisation. Section extraction parses legal numbering conventions into structured JSON. The stratified design enables length-sensitivity analysis, which proves critical for understanding LC failure modes (Section 4.5).

All 50 sampled verdicts were embedded using Google text-embedding-004 with 512-token fixed-size chunking (50-token overlap) and indexed with FAISS (Johnson et al., 2021) for dense retrieval.

### 3.2 QA Dataset Construction

A dataset of **300 question-answer pairs** was constructed via a three-stage pipeline: LLM-assisted draft generation, mandatory human review, and automated consistency validation.

Draft generation used Gemini 2.5 Flash to propose candidate questions across four cognitive types: factual extractive (100 pairs), multi-section reasoning (100 pairs), structural (50 pairs), and boundary (50 pairs). All 300 drafts were individually reviewed via a custom CLI applying an accept/modify/reject protocol. Gold answers were required to be verifiable from explicit source text, constraining the space of legitimate disagreement.

Two automated consistency checks were applied before annotation. A verbatim grounding checker flagged 8 of 300 pairs (2.7%) with genuine grounding failures for human correction. A semantic duplicate detector using cosine similarity identified 28 candidate pairs with similarity ≥ 0.88, of which 7 were confirmed duplicates removed from the initial generation pool; replacement pairs were generated to maintain the target balance, yielding a final benchmark of **300 pairs**.

**Inter-annotator agreement.** A second annotator independently reviewed a randomly drawn 10% overlap set of 30 QA pairs, blind to the primary annotator's labels. We report observed agreement, Cohen's kappa, and the label distribution together, because kappa alone is misleading on this task.

| Metric | Value |
|:---|---:|
| Shared items | 30 |
| Observed agreement | 0.900 |
| Cohen's κ | 0.000 |
| Annotator 1: `accepted` | 27 |
| Annotator 1: `modified` | 3 |
| Annotator 2: `accepted` | 30 |

The gap between 0.900 observed agreement and κ = 0.000 is a textbook instance of the prevalence-kappa paradox (Feinstein and Cicchetti, 1990). When one label dominates, the chance-agreement baseline approaches the observed-agreement ceiling, collapsing kappa toward zero even when annotators are substantively consistent. Both annotators agreed on 27 of 30 items outright; the 3 disagreements reflect minor phrasing judgements rather than factual disputes. We treat 0.900 observed agreement as the primary reliability signal and report kappa as a transparency measure, following the reporting convention recommended by Feinstein and Cicchetti (1990).

### 3.3 Architectures

**Long Context (LC)** injects the full verdict text directly into the generation prompt. For Gemini 2.5 Flash this enables inputs exceeding 50,000 tokens per query; for GPT-4o Mini the 128,000-token window imposes practical truncation on the longest verdicts.

**Simple RAG** chunks each verdict into 512-token segments with 50-token overlap, embeds chunks with text-embedding-004, and retrieves the top-5 most similar chunks via FAISS cosine similarity (Johnson et al., 2021). Retrieved chunks are concatenated into the generation prompt.

**Advanced RAG** augments Simple RAG with four optional components evaluated both individually (ablation study) and in combination: LLM query rewriting, BM25+dense hybrid search with Reciprocal Rank Fusion (Cormack et al., 2009), cross-encoder reranking (ms-marco-MiniLM-L-6-v2; Nogueira and Cho, 2019; Reimers and Gurevych, 2019), and metadata filtering by verdict court and year.

### 3.4 Evaluation

The primary evaluation metric is **faithfulness**, computed via a two-stage judge pipeline. Gemini 3 Flash Preview performs both answer decomposition into atomic factual claims and per-claim grounding evaluation. When the primary model returns a parseable JSON verdict directly, no second call is made; when its response is free text, a Gemini 2.5 Flash Lite call with a constrained JSON schema extracts the verdict. This fallback design eliminates JSON-parsing failures observed with single-model judges on complex Indonesian legal text while keeping the strong reasoning model in the critical evaluation role.

Secondary metrics include: hallucination rate (HR = 1 − faithfulness), BERTScore F1 computed with IndoBERT (Koto et al., 2020; Zhang et al., 2020), context precision and recall (RAG conditions), input token cost, and wall-clock latency. Statistical significance for main comparisons uses paired one-sided Wilcoxon signed-rank tests with Bonferroni correction. Effect sizes are reported as Cohen's *d* with bootstrap 95% confidence intervals (5,000 iterations).

A 10% human legal accuracy spot-check (0/1/2 scale) was conducted on Phase 1 outputs and is reported in Section 6.

### 3.5 Faithfulness Evaluation Correction

A post-submission audit identified a logging bug in the Long Context implementation: `LongContextSystem` stored a 503-character truncation preview (`text[:500] + "..."`) in the `context_used` field rather than the actual generation context. The experiment runner passed `context_used` directly to `evaluate_faithfulness`, so all LC faithfulness scores in the original analysis were evaluated against the document beginning, not the context used for generation. For long verdicts averaging over 900,000 characters, the preview covered less than 0.1% of each document.

To produce correct and comparable faithfulness figures across all three architectures, we re-evaluated all saved answers — LC, Simple RAG, and Advanced RAG — against the `gold_paragraphs` field: human-verified evidence paragraphs confirmed during QA dataset construction. This oracle metric (**gold-evidence faithfulness**) tests whether each answer is supported by the known-correct evidence, independent of retrieval quality or context injection method. All faithfulness, hallucination rate, and derived statistics in Sections 4.1–4.6 reflect this corrected evaluation. The `context_used` storage bug has been fixed in the published codebase. A full correction note with before/after tables is available at `CORRECTION_NOTE.md` in the repository.

---

## 4. Results

### 4.1 Phase 1 Baseline

Table 1 presents Phase 1 results using Gemini 2.5 Flash across all three architectures on the 300-question benchmark. The LC condition experienced 51 quota-exhaustion errors (RESOURCE_EXHAUSTED) due to extreme payload sizes; faithfulness is reported over the 249 successful queries.

**Table 1. Phase 1 baseline (gold-evidence faithfulness).**

| Architecture | Queries | Faithfulness | BERTScore F1 | Halluc. Rate | Zero-faith | Tokens (mean) | Cost (USD) | Latency |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Simple RAG | 300 | **0.508** | **0.510** | 0.492 | 107 | 1,990 | **$0.31** | 17.9 min |
| Advanced RAG | 300 | 0.481 | 0.490 | 0.519 | 119 | 1,339 | $0.22 | 30.0 min |
| Long Context | 249* | 0.599*† | 0.459‡ | 0.401*† | 64*† | 29,789‡ | $4.48 | 14.7 min |

*51 records contain quota-exhaustion errors. Phase 2 provides complete 300-query LC runs and serves as the primary LC comparison.
†LC faithfulness and hallucination rate are scored-only (249 answered queries). Coverage-adjusted (51 non-answers = 0 faithfulness): mean = **0.497**, halluc. rate = 0.503, zero-faith = 115.
‡BERTScore F1 and mean input tokens for LC are coverage-adjusted. Scored-only: BERTScore F1 = 0.553, mean tokens = 35,891.

The 51 LC failures are not random. Cross-referencing with stratum labels in `sample_50.csv` shows all 51 occur in the long-verdict stratum: a 56.7% failure rate for long-stratum queries (51/90) against 0% for short and medium verdicts combined (0/210). The architecture degrades operationally exactly where long-context prompting should be most advantageous.

Among the 249 answered LC queries, LC achieves gold-evidence faithfulness 0.599, significantly exceeding Simple RAG's 0.508 on those same pairs (d = 0.193, p = 1.17e-3). Coverage-adjusted — treating the 51 non-answers as zero faithfulness — Simple RAG (0.508) marginally leads LC (0.497), a gap of 0.011 attributable entirely to the operational non-response rate on long-verdict queries, not faithfulness quality.

### 4.2 Phase 2 Multi-Model Factorial

Phase 2 provides the primary architectural comparison because all six cells are complete 300-query runs.

**Table 2. Phase 2 full factorial results with bootstrap 95% CIs (gold-evidence faithfulness).**

| Model | Architecture | Queries | Faithfulness | 95% CI | BERTScore F1 | Halluc. Rate | Zeros |
|:---|:---|---:|---:|:---:|---:|---:|---:|
| Gemini 2.5 Flash | Simple RAG | 300 | 0.557 | [0.507, 0.606] | 0.507 | 0.443 | 98 |
| Gemini 2.5 Flash | Advanced RAG | 300 | 0.454 | [0.406, 0.504] | 0.489 | 0.546 | 131 |
| Gemini 2.5 Flash | Long Context | 300 | **0.561** | [0.514, 0.609] | **0.548** | **0.439** | **90** |
| GPT-4o Mini | Simple RAG | 300 | 0.565 | [0.511, 0.620] | 0.681 | 0.435 | 119 |
| GPT-4o Mini | Advanced RAG | 300 | 0.476 | [0.422, 0.528] | 0.647 | 0.524 | 146 |
| GPT-4o Mini | Long Context | 300 | **0.575** | [0.522, 0.628] | **0.728** | **0.425** | **118** |

LC faithfulness is bolded as numerically highest; the difference from Simple RAG is not statistically significant (see Table 2A).

**Table 2A. Phase 2 paired effect sizes (gold-evidence faithfulness).**

| Model | Comparison | Mean diff. | Cohen's *d* | 95% CI | *p*-value |
|:---|:---|---:|---:|:---:|---:|
| Gemini 2.5 Flash | LC vs Simple RAG | +0.004 | 0.008 | [−0.107, 0.122] | 0.434 (n.s.) |
| Gemini 2.5 Flash | LC vs Advanced RAG | +0.107 | 0.202 | [0.090, 0.319] | 1.55e-4 |
| Gemini 2.5 Flash | Simple RAG vs Advanced RAG | +0.103 | 0.188 | [0.070, 0.305] | 6.15e-4 |
| GPT-4o Mini | LC vs Simple RAG | +0.010 | 0.017 | [−0.095, 0.131] | 0.489 (n.s.) |
| GPT-4o Mini | LC vs Advanced RAG | +0.099 | 0.175 | [0.065, 0.291] | 3.04e-3 |
| GPT-4o Mini | Simple RAG vs Advanced RAG | +0.089 | 0.160 | [0.047, 0.277] | 1.96e-3 |

Under gold-evidence faithfulness, the architecture ordering is **LC ≈ Simple RAG >> Advanced RAG** across both model families. The LC vs Simple RAG difference is not statistically significant for either Gemini Flash (d = 0.008, p = 0.43) or GPT-4o Mini (d = 0.017, p = 0.49), indicating the two architectures achieve equivalent faithfulness under oracle evaluation. Both LC and Simple RAG significantly outperform Advanced RAG: Gemini Flash p(LC > AR) = 1.55e-4, p(SR > AR) = 6.15e-4; GPT-4o Mini p(LC > AR) = 3.04e-3, p(SR > AR) = 1.96e-3. The original d = 0.582–0.803 reported for Simple RAG vs Long Context was an artefact of the evaluation bug.

Effect sizes (Table 2A) confirm that the LC vs Simple RAG separation is negligible (d = 0.008–0.017), while the advantage over Advanced RAG is small but consistent (d = 0.160–0.202).

A notable measurement asymmetry: Long Context consistently achieves the highest BERTScore F1 across all conditions. For GPT-4o Mini, LC reaches BERTScore 0.728 while gold-evidence faithfulness is 0.575 — slightly above Simple RAG (BERTScore 0.681, faithfulness 0.565). LC answers tend to be longer and semantically closer to reference answers at the surface level. For legal QA, BERTScore alone is insufficient as a primary metric; faithfulness against authoritative evidence is a more direct measure of factual grounding.

### 4.3 Cost and Latency

**Table 3. Phase 2 efficiency summary.**

| Model | Architecture | Tokens (mean) | Cost/query (USD) | Total cost (USD) | Latency |
|:---|:---|---:|---:|---:|---:|
| Gemini Flash | Simple RAG | 1,990 | 0.001044 | 0.3131 | 18.1 min |
| Gemini Flash | Advanced RAG | 1,324 | 0.000710 | 0.2130 | 31.5 min |
| Gemini Flash | Long Context | 51,719 | 0.025913 | 7.7740 | 18.2 min |
| GPT-4o Mini | Simple RAG | 2,332 | 0.000210 | 0.0630 | 23.6 min |
| GPT-4o Mini | Advanced RAG | 1,550 | **0.000146** | **0.0438** | 38.3 min |
| GPT-4o Mini | Long Context | 45,051 | 0.003421 | 1.0262 | 69.6 min |

Under Gemini Flash, Simple RAG is 24.8× cheaper than Long Context while achieving higher faithfulness. Under GPT-4o Mini, 16.3× cheaper. The cheapest condition overall is GPT-4o Mini + Advanced RAG ($0.0438); the most expensive is Gemini Flash + Long Context ($7.7740) — a 177× cost spread across tested configurations. The most expensive configuration is not the best-performing one.

### 4.4 Ablation Study

**Table 4. Ablation study (100-question subset, Gemini Flash, gold-evidence faithfulness).**

| Condition | Faithfulness | BERTScore F1 | Halluc. Rate | Latency/q | Cost/q (USD) |
|:---|---:|---:|---:|---:|---:|
| Baseline Simple RAG ⭐ | **0.604** | 0.513 | **0.396** | 3.81 s | 0.001037 |
| + Query Rewrite | 0.453 ↓↓ | 0.482 | 0.547 ↑↑ | 5.76 s | 0.000964 |
| + Metadata Filter | 0.577 ↓ | 0.507 | 0.423 ↑ | 3.82 s | 0.001035 |
| + Hybrid Search | 0.585 ↓ | **0.533** | 0.415 ↑ | **3.75 s** | 0.001174 |
| + Reranking | 0.490 ↓↓ | 0.497 | 0.510 ↑↑ | 4.18 s | **0.000700** |
| Full Advanced RAG | 0.472 ↓↓ | 0.526 | 0.528 ↑↑ | 6.26 s | 0.000710 |

Under gold-evidence faithfulness, the ablation hierarchy reverses from the original retrieved-chunk analysis: **Baseline Simple RAG** ranks first (0.604), and every augmentation component reduces oracle faithfulness. Hybrid BM25+dense search is the least harmful augmentation (0.585, −1.9 pp from baseline), followed by metadata filtering (0.577, −2.8 pp). Query rewriting is the most damaging step under this metric (0.453, −15.1 pp), followed by Full Advanced RAG (0.472, −13.3 pp) and reranking (0.490, −11.4 pp).

The reversal is methodologically significant: retrieved-chunk faithfulness and gold-evidence faithfulness measure different things. Hybrid search improves recall of gold-relevant passages, boosting chunk-faithfulness in the original evaluation (+9.2 pp) while not necessarily grounding answers in the exact human-verified gold paragraph text. Under oracle evaluation, the baseline concentrates generation on the most semantically similar chunks, producing the tightest correspondence to authoritative evidence. Practitioners should validate augmentation choices against oracle evidence, not only chunk-level similarity.

The reranking finding is consistent across both metrics (−9.0 pp original, −11.4 pp gold-evidence). The ms-marco-MiniLM-L-6-v2 cross-encoder (Nogueira and Cho, 2019; Reimers and Gurevych, 2019) was trained on English MS MARCO data and has no exposure to Indonesian legal prose. It likely overweights superficial lexical similarity while underweighting domain-specific cues such as legal article citations, court terminology, and section-local procedural language. Because reranking also prunes aggressively, a moderate ranking error becomes an unrecoverable evidence loss.

### 4.5 Length Sensitivity

Notebook 07 presents Phase 1 gold-evidence faithfulness broken down by verdict-length stratum.

**Table 5. Phase 1 gold-evidence faithfulness by verdict-length stratum.**

| Architecture | Short | Medium | Long (scored-only) | Long (coverage-adj.) |
|:---|---:|---:|---:|---:|
| Simple RAG | 0.483 | 0.527 | **0.507** | **0.507** |
| Advanced RAG | 0.439 | 0.503 | 0.494 | 0.494 |
| Long Context | **0.578** | **0.606** | 0.628 | 0.272* |

*Coverage-adjusted: 51 of 90 long-stratum LC queries returned no answer (quota exhaustion). Coverage-adjusted = 39/90 × 0.628 = 0.272.

Under gold-evidence evaluation, LC faithfulness quality **increases** monotonically with verdict length (0.578 → 0.606 → 0.628 among answered queries) — the opposite of the original finding. Simple RAG and Advanced RAG also improve slightly with length. The original pattern (LC: 0.448 → 0.533 → 0.205) was entirely caused by the evaluation bug: for long verdicts averaging over 900,000 characters, the 503-character logging preview used in the original faithfulness evaluation covered less than 0.1% of each document.

The actual long-verdict failure is **operational, not qualitative**. Among 90 long-stratum LC queries, 51 (56.7%) returned no answer due to quota exhaustion — against 0% for short and medium verdicts. Among the 39 answered long-stratum queries, LC faithfulness is 0.628 — the highest of any architecture in any stratum. Coverage-adjusted (treating 51 failures as zero faithfulness), LC scores 0.272, well below Simple RAG's 0.507 on the same stratum. The gap reflects non-response, not faithfulness quality when the system does respond. This degradation is paired with a token explosion: mean input tokens grow from 11.1k (short) to 23.4k (medium) to 131.4k (long) for LC. Document length increases the likelihood of quota exhaustion, not the likelihood of faithfulness failure in completed answers.

### 4.6 Needle-in-a-Haystack Evaluation

**Table 6. Needle-in-a-haystack results (30 queries, gold evidence at depth ≥ 80%).**

| Architecture | Queries | NIAH Accuracy | Faithfulness | BERTScore F1 | Zero-faith | Latency |
|:---|---:|---:|---:|---:|---:|---:|
| Simple RAG | 30 | **0.8333** | 0.489 | 0.5058 | 10 | 1.80 min |
| Advanced RAG | 30 | 0.7667 | 0.467 | 0.5075 | 12 | 3.19 min |
| Long Context | 30 | 0.5000 | **0.578** | **0.5476** | **8** | **1.68 min** |

NIAH accuracy (derived from human-evaluated `legal_accuracy` scores) is unchanged by the faithfulness correction. Simple RAG retrieves and grounds deep evidence effectively, answering 29 of 30 queries correctly on the human accuracy measure (0.833). Under gold-evidence faithfulness, LC leads all three architectures (0.578 vs 0.489 for Simple RAG), indicating that LC answers to deep-evidence NIAH queries are well-grounded in known-correct evidence when they are produced. LC succeeds on only 15 of 30 queries by human accuracy (0.500), consistent with the "lost in the middle" effect (Liu et al., 2023): the accuracy failure is one of task completion — the model fails to locate and answer from the needle passage — not of faithfulness quality in completed answers. Retrieval's signal concentration advantage persists on accuracy even when faithfulness quality is comparable.

---

## 5. Discussion

### 5.1 Why Simple RAG Is Operationally More Reliable

Under gold-evidence evaluation, Simple RAG and Long Context achieve statistically equivalent faithfulness quality in Phase 2. Simple RAG's primary advantage is operational: 100% response coverage at a fraction of LC's cost, with no quota-exhaustion failures on any stratum. The most plausible explanation for the LC faithfulness tie — despite different generation contexts — is that both architectures are evaluated against the same oracle evidence, removing the retrieval-quality confound. When answers are produced, LC has access to the full document and naturally produces answers grounded in gold evidence; Simple RAG achieves the same by concentrating the prompt on the retrieved relevant chunks.

This interpretation is supported by four converging patterns: (1) LC and Simple RAG achieve statistically indistinguishable faithfulness on Phase 2 complete runs; (2) LC faithfulness quality is highest on long verdicts among answered queries (0.628), suggesting the full-document context is not suppressing faithfulness when the model completes the task; (3) NIAH accuracy strongly favours retrieval (0.833 vs 0.500), precisely where task-completion, not faithfulness quality, is the critical metric; and (4) LC operational reliability collapses on the longest verdicts (56.7% non-response), while faithfulness quality among answered queries is highest — separating a resource-throttling failure from an attention-based failure.

### 5.2 Why More RAG Is Not Always Better

The ablation shows that additional retrieval components introduce failure modes of their own. Under gold-evidence evaluation, every augmentation component reduces oracle faithfulness relative to baseline. Query rewriting likely broadens or distorts the original information need, producing the largest oracle faithfulness drop (−15.1 pp). The reranker is especially brittle because it is not adapted to Indonesian legal language and prunes aggressively — a moderate ranking error becomes an unrecoverable evidence loss (−11.4 pp oracle). Hybrid search and metadata filtering reduce oracle faithfulness slightly (−1.9 pp and −2.8 pp respectively), suggesting modest but consistent over-retrieval.

The lesson for practitioners is not that Advanced RAG is bad, but that retrieval pipelines must be validated against authoritative evidence, not only retrieved-chunk overlap. An English-trained cross-encoder should not be assumed to transfer to Bahasa Indonesia legal prose, and augmentation components that improve retrieval recall may not improve answer grounding against known-correct evidence.

### 5.3 BERTScore as a Secondary Metric

BERTScore (Zhang et al., 2020) consistently ranks Long Context highest across all conditions — a pattern consistent with both the original and corrected evaluations. LC answers are verbose and semantically close to reference answers at the surface level. Under gold-evidence faithfulness, this surface similarity is matched by genuine grounding quality (LC leads or ties Simple RAG). The original divergence between high BERTScore and low faithfulness was artefactual: the evaluation bug produced low faithfulness scores while BERTScore was computed independently. Researchers evaluating legal QA systems should use faithfulness — computed via a claim-grounding judge — as the primary metric for factual grounding, with BERTScore as a complementary signal for lexical and semantic coverage.

### 5.4 Operational Reliability

The Phase 1 LC failures are not a theoretical concern: all 51 occur on long-verdict queries, corresponding to a 56.7% failure rate in that stratum. Even in Phase 2 where LC runs complete successfully, the pattern is informative for production deployment. Systems handling repeated long-document workloads face quota and throughput constraints that retrieval-based systems avoid by design.

---

## 6. Human Legal Accuracy Validation

To validate the automated faithfulness judge against human legal expertise, a 10% sample of Phase 1 answers was reviewed by a human evaluator with legal domain knowledge, using a three-point scale:

- **0** — factually wrong or unsupported by the source verdict
- **1** — partially correct but incomplete or imprecise
- **2** — legally accurate and fully grounded in the source text

Approximately 30 answers per condition (LC, Simple RAG, Advanced RAG) were evaluated, totalling approximately 90 judgements.

**Table 7. Human legal accuracy spot-check (Phase 1, filtered sample).**

| Architecture | Mean Legal Accuracy (0–2) |
|:---|---:|
| Long Context | **1.80** |
| Simple RAG | 1.73 |
| Advanced RAG | 1.47 |

The human ranking (LC > Simple RAG > Advanced RAG) is now consistent with the corrected automated gold-evidence faithfulness ranking (LC ≈ Simple RAG > Advanced RAG) and must still be interpreted with care. The review workflow excluded obviously truncated outputs, which systematically favours Long Context's *surviving* completions: in the full Phase 1 run, Long Context incurred 51 unscored quota-exhaustion failures concentrated entirely in the long-verdict stratum. Truncated or absent answers were not represented in the human review sample. The spot-check is best read as evidence that *successful* Long Context answers tend to be legally coherent prose — consistent with the corrected finding that LC faithfulness is highest among answered queries. The automated faithfulness judge is validated for claim-level grounding; the human review adds nuance about prose accuracy and legal precision among completed answers.

---

## 7. Limitations

**Domain specificity.** Results are grounded in Indonesian Constitutional Court PUU-type verdicts and may not generalise to other legal corpora, languages, or verdict types.

**Knowledge-update evidence.** A three-verdict knowledge-update pilot is directionally consistent with the main findings (Simple RAG: 1.000, Long Context: 0.667, Advanced RAG: 0.333) but n = 3 is too small to support a paper-level claim. An expanded experiment with n ≥ 15 is planned for future work.

**Inferential scope.** Main Phase 2 comparisons report bootstrap confidence intervals, Wilcoxon tests, and Cohen's *d*, but no family-wise correction is applied across all exploratory pairwise comparisons.

**Human legal validation.** A 10% human legal accuracy spot-check was conducted on Phase 1 outputs (Section 6). The human ranking of surviving answers (LC > Simple RAG > Advanced RAG) is consistent with the corrected automated faithfulness ranking (LC ≈ Simple RAG > Advanced RAG). However, truncated LC outputs were excluded from human review, and the spot-check cannot account for the 51 non-responses. Both findings are reported transparently alongside their respective sampling caveats.

**Inter-annotator agreement.** The 30-item overlap set yields 0.900 observed agreement alongside κ = 0.000. As discussed in Section 3.2, this gap is fully explained by the near-uniform `accepted` label distribution across both annotators (Feinstein and Cicchetti, 1990), not by genuine rater disagreement.

**Reranker domain transfer.** The ms-marco cross-encoder failure (−9.0 pp original retrieved-chunk faithfulness; −11.4 pp gold-evidence faithfulness) indicates that retrieval components trained on English web-scale data do not reliably transfer to Indonesian legal prose. Future work should develop an Indonesian legal domain-specific reranker.

---

## 8. Conclusion

This study provides the first systematic empirical comparison of Long Context, Simple RAG, and Advanced RAG architectures on Indonesian Constitutional Court verdicts, and identifies and corrects a faithfulness evaluation bug that substantially understated LC performance in the original analysis. Under corrected oracle (gold-evidence) evaluation, Long Context achieves faithfulness statistically indistinguishable from Simple RAG in Phase 2; the principal advantage of targeted retrieval is cost (24.8× cheaper under Gemini Flash) and operational reliability, not faithfulness quality.

The most practically important finding is the LC operational reliability collapse on long verdicts: 56.7% non-response rate despite LC faithfulness quality being highest in that stratum (0.628 among answered queries). That LC answers are well-grounded when produced, but frequently not produced at all, points to resource-throttling as the key failure mode — a finding relevant to any practitioner deploying long-context LLMs on large legal documents.

On ablation, Baseline Simple RAG achieves the highest gold-evidence faithfulness (0.604), with every augmentation reducing it. The reranking failure (−11.4 pp gold-evidence, −9.0 pp original) is a domain-adaptation warning relevant to all Indonesian legal NLP practitioners using English-trained retrieval components.

The practical implication is that context capacity is not irrelevant, but that it should be applied after retrieval has concentrated the relevant evidence — and that practitioners must ensure quota headroom for long-document workloads. Faithfulness quality alone does not distinguish the architectures; cost, latency, and operational completeness do. The benchmark corpus, QA dataset, and full pipeline are publicly available at github.com/Shiverion/VerdictBench-LCvsRAG.

---

## References

- Cormack, G. V., Clarke, C. L. A., and Buettcher, S. (2009). Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. *SIGIR 2009*, 758–759.
- Es, S., James, J., Espinosa-Anke, L., and Schockaert, S. (2024). RAGAS: Automated Evaluation of Retrieval Augmented Generation. *arXiv:2309.15217*.
- Feinstein, A. R. and Cicchetti, D. V. (1990). High agreement but low kappa: I. The problems of two paradoxes. *Journal of Clinical Epidemiology*, 43(6), 543–549.
- Guha, N., et al. (2023). LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models. *NeurIPS*.
- Guu, K., Lee, K., Tung, Z., Pasupat, P., and Chang, M. (2020). REALM: Retrieval-Augmented Language Model Pre-Training. *ICML*.
- Hendrycks, D., et al. (2021). CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review. *NeurIPS*.
- Johnson, J., Douze, M., and Jégou, H. (2021). Billion-Scale Similarity Search with GPUs. *IEEE Transactions on Big Data*, 7(3), 535–547.
- Koto, F., Rahimi, A., Lau, J. H., and Baldwin, T. (2020). IndoLEM and IndoBERT: A Benchmark Dataset and Pre-trained Language Model for Indonesian NLP. *COLING*.
- Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.
- Li, T., Zhang, G., Do, Q. D., Yue, X., and Chen, W. (2024). Long-context LLMs Struggle with Long In-context Learning. *Transactions on Machine Learning Research (TMLR)*. arXiv:2404.02060.
- Liu, N. F., et al. (2023). Lost in the Middle: How Language Models Use Long Contexts. *Transactions of the Association for Computational Linguistics (TACL)*, 12, 157–173.
- Nogueira, R. and Cho, K. (2019). Passage Re-ranking with BERT. *arXiv:1901.04085*.
- Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP*. arXiv:1908.10084.
- Robertson, S. and Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*.
- Xu, P., Ping, W., Wu, X., McAfee, L., Zhu, C., Liu, Z., Subramanian, S., Bakhturina, E., Shoeybi, M., and Catanzaro, B. (2024). Retrieval Meets Long Context Large Language Models. *ICLR 2024*. arXiv:2310.03025.
- Zhang, T., et al. (2020). BERTScore: Evaluating Text Generation with BERT. *ICLR*.

---

## Appendix A. Artifact Grounding

All quantitative claims in this paper are grounded in the following committed repository artifacts:

- `results/phase1/simple_rag/run_20260320_080919_bs.jsonl`
- `results/phase1/advanced_rag/run_20260320_164547_bs.jsonl`
- `results/phase1/lc/run_20260320_073844_bs.jsonl`
- `results/phase2/gemini_flash/*/results_clean_bs.jsonl`
- `results/phase2/gpt4o/*/results_clean_bs.jsonl`
- `results/ablation/*/results_clean_bs.jsonl`
- `results/additional/niah/niah_summary.csv`
- `data/metadata/sample_50.csv`
- `data/qa_dataset/iaa_existing_overlap/kappa_summary.json`
- `results_corrected/gold_evidence_faithfulness/*.jsonl`