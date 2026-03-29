# Notebook Key Findings

This note condenses the executed outputs from `notebooks/00` through `notebooks/10` into the few findings that materially matter for readers. It is intentionally narrower than [`report.md`](C:\Users\miqba\projects\LC vs RAG benchmark\report.md): the goal here is to surface the strongest empirical signals, not to reproduce the full paper narrative.

## 1. Dataset And Benchmark Integrity

- The benchmark is built from **50 Indonesian Constitutional Court verdicts** sampled from a larger audited corpus and stratified by length: **15 short, 20 medium, 15 long**.
- The final QA benchmark contains **300 human-reviewed question-answer pairs**.
- Question-type distribution is balanced toward realistic legal QA:
  - **100 factual extractive**
  - **100 multi-section reasoning**
  - **50 structural**
  - **50 boundary**
- The overlap annotation check on **30 shared items** shows **0.900 observed agreement**. Cohen's kappa is **0.000**, but this is a prevalence artifact, not true annotation collapse: one label dominates the overlap set, so kappa is not the main reliability signal here.

## 2. Main Empirical Result

Across complete comparisons, the architecture ranking is stable:

**Simple RAG > Advanced RAG > Long Context**

This holds in both the Phase 1 baseline and the Phase 2 multi-model evaluation.

### Phase 1 Baseline

| Architecture | Scorable queries | Mean faithfulness | Mean BERTScore F1 | Total cost (USD) |
|:---|---:|---:|---:|---:|
| Simple RAG | 300 | **0.845** | **0.510** | **0.3138** |
| Advanced RAG | 300 | 0.764 | 0.490 | 0.2153 |
| Long Context | 249 | 0.451 | 0.459 | 4.4824 |

Key takeaway:
- Simple RAG is the strongest Phase 1 system on faithfulness and cost.
- Long Context is worst by a large margin and does not compensate with better grounding.

### Phase 2 Multi-Model Evaluation

| Model | Architecture | Mean faithfulness | Mean BERTScore F1 | Mean hallucination rate |
|:---|:---|---:|---:|---:|
| Gemini 2.5 Flash | Simple RAG | **0.840** | 0.507 | **0.160** |
| Gemini 2.5 Flash | Advanced RAG | 0.762 | 0.489 | 0.238 |
| Gemini 2.5 Flash | Long Context | 0.616 | **0.548** | 0.384 |
| GPT-4o Mini | Simple RAG | **0.857** | 0.681 | **0.143** |
| GPT-4o Mini | Advanced RAG | 0.780 | 0.647 | 0.220 |
| GPT-4o Mini | Long Context | 0.524 | **0.728** | 0.476 |

Key takeaways:
- The ranking is unchanged across **Gemini 2.5 Flash** and **GPT-4o Mini**.
- **Long Context has the highest BERTScore but the lowest faithfulness** in both model families.
- On this legal QA benchmark, semantic similarity alone would overstate Long Context quality.

## 3. The Long-Context Failure Is Both Quality And Reliability

The Long Context result is not just lower quality. It is also less operationally reliable.

- In Phase 1, **51 LC records are unscored** because of Gemini `429 RESOURCE_EXHAUSTED` failures.
- Those failures are not random.
- They are concentrated entirely in the **long-verdict stratum**:
  - **51 / 90 long-stratum queries failed**
  - **0 / 210 short+medium queries failed**

Important implication:
- Long-context prompting degrades exactly where it is supposed to help most, namely on the longest documents.

Notebook 07 adds the missing stratified quality view behind that failure pattern. In Phase 1 faithfulness by verdict-length stratum:

| Architecture | Short | Medium | Long |
|:---|---:|---:|---:|
| Simple RAG | **0.869** | **0.859** | **0.803** |
| Advanced RAG | 0.765 | 0.796 | 0.721 |
| Long Context | 0.448 | 0.533 | **0.205** |

This is the more important interpretation of the length-sensitivity notebook:
- **Simple RAG stays relatively stable** as verdict length increases.
- **Long Context collapses in the long stratum**, where its mean faithfulness falls to **0.205**.
- The LC drop is paired with a token explosion:
  - short: **11.1k** mean input tokens
  - medium: **23.4k**
  - long: **131.4k**

So the length-sensitive result is not merely that LC sometimes fails. It is that longer verdicts make LC both less reliable and less faithful, while retrieval-based systems remain comparatively robust.

## 4. Effect Sizes Confirm The Main Comparison

Phase 2 significance is not only statistical; it is practically meaningful, especially against Long Context.

| Model | Comparison | Cohen's d | 95% bootstrap CI |
|:---|:---|---:|:---:|
| Gemini 2.5 Flash | Simple RAG vs Advanced RAG | 0.220 | [0.065, 0.379] |
| Gemini 2.5 Flash | Simple RAG vs Long Context | 0.582 | [0.411, 0.753] |
| GPT-4o Mini | Simple RAG vs Advanced RAG | 0.206 | [0.049, 0.361] |
| GPT-4o Mini | Simple RAG vs Long Context | 0.803 | [0.629, 0.988] |

Interpretation:
- **Simple RAG vs Advanced RAG** is a small but consistent gain.
- **Simple RAG vs Long Context** is a medium-to-large gain.
- The empirical gap against Long Context is therefore not just a marginal win.

## 5. Retrieval Is Also Cheaper

The more faithful system is also the cheaper one.

| Model | Simple RAG total cost | Long Context total cost | Relative difference |
|:---|---:|---:|---:|
| Gemini 2.5 Flash | 0.3131 | 7.7740 | **24.8x cheaper** |
| GPT-4o Mini | 0.0630 | 1.0262 | **16.3x cheaper** |

Across all Phase 2 conditions, the total-cost spread is roughly **177x** from the cheapest to the most expensive setting, and the most expensive setting is not the best one.

## 6. The Best Advanced-RAG Component Is Hybrid Search

The ablation notebooks show that the full Advanced RAG stack is not uniformly helpful. One component is clearly beneficial, while two are clearly harmful.

| Condition | Mean faithfulness |
|:---|---:|
| Baseline Simple RAG | 0.815 |
| + Query Rewrite | 0.767 |
| + Metadata Filter | 0.840 |
| + Hybrid Search | **0.907** |
| + Reranking | 0.725 |
| Full Advanced RAG | 0.803 |

Key takeaways:
- **Hybrid search is the single best ablation condition**.
- **Metadata filtering helps**.
- **Query rewriting hurts** in its current form.
- **Cross-encoder reranking hurts the most** and appears to be the main reason the full Advanced RAG pipeline trails Simple RAG.

The most plausible explanation is domain mismatch: the reranker is optimized for general-domain ranking signals rather than Bahasa Indonesia legal evidence selection.

## 7. Needle-In-A-Haystack Still Favors Retrieval

The NIAH notebook asks a focused stress-test question: when gold evidence is buried deep in the verdict, does direct full-document prompting recover it better than retrieval?

No.

| Architecture | NIAH accuracy | Mean faithfulness |
|:---|---:|---:|
| Simple RAG | **0.8333** | **0.9056** |
| Advanced RAG | 0.7667 | 0.8167 |
| Long Context | 0.5000 | 0.6833 |

Key takeaway:
- Even under deep-evidence conditions, **retrieval remains stronger than full-document prompting**.

## 8. Bottom Line

The notebook suite supports one clear conclusion:

> On Indonesian Constitutional Court verdict QA, targeted retrieval is more faithful, cheaper, and more operationally robust than injecting the entire verdict into the prompt.

The strongest single practical recommendation from the notebooks is:

- start from **Simple RAG**
- add **metadata filtering** and especially **hybrid search**
- avoid assuming that a large context window makes retrieval unnecessary

## 9. Remaining Gap

One submission-relevant item is still outside the completed notebook evidence:

- **Human legal accuracy spot-check**

The automated faithfulness results are already strong, but a manual legal-accuracy validation pass would further strengthen the paper for peer review.
