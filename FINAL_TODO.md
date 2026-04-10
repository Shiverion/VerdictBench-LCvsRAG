# Final To-Do

This file is now a **post-close submission status note**. The main paper-facing items from the final push have been completed.

## Final Status

### 1. Human Legal Accuracy Spot-Check

**Status:** done  
**Outcome:** a 30-item-per-condition human legal-accuracy spot-check was completed on Phase 1 outputs using the 0/1/2 rubric.  
**Observed means on filtered, non-obviously-truncated outputs:** Long Context `1.80`, Simple RAG `1.73`, Advanced RAG `1.47`.

**Interpretation**

- This human spot-check should **not** be read as a full condition-level ranking.
- The review workflow excluded obviously truncated outputs, which favors Long Context's surviving completions.
- In the full Phase 1 run, Long Context still incurred `51` unscored failures, all in the long-verdict stratum.
- The paper therefore reports the human review as evidence that surviving LC answers can be legally strong, while the full-run architecture comparison remains better reflected by faithfulness, reliability, and cost.

**Artifacts**

- `results/phase1/simple_rag/*_human.jsonl`
- `results/phase1/advanced_rag/*_human.jsonl`
- `results/phase1/lc/*_human.jsonl`
- `results/legal_accuracy_summary.md`

---

## Completed in This Pass

These items were pending previously and are now done in the paper/docs:

1. **Phase 2 effect sizes.** `report.md` now includes Cohen's *d* with bootstrap confidence intervals for all main Phase 2 pairwise comparisons.
2. **Knowledge-update decision.** The `n = 3` pilot has been demoted from the main empirical claim and is treated as exploratory only.
3. **Phase 1 LC failure analysis by stratum.** The report now states that all 51 Phase 1 LC failures occur in the `long` stratum.
4. **BERTScore coverage confirmation.** The report now states that all scorable records in the main Phase 1, Phase 2, and ablation tables have populated `bertscore_f1` values.
5. **Abstract trim.** The abstract in `report.md` has been reduced and no longer leans on the underpowered knowledge-update pilot.
6. **Title unification.** `README.md`, `report.md`, and the citation title now use the same paper title.
7. **Human legal spot-check.** The paper now includes the filtered-sample human legal-accuracy result with the required selection-bias caveat.

---

## Optional Follow-Up Polish

These are not blocking after the current pass, but they would still improve the repo:

- Clean the remaining README formatting artifacts near the top section if you want a fully polished public landing page.
- If you want a revision-round upgrade, rerun the human spot-check without filtered-sample bias by separately tracking truncated completions as zero-quality outputs.
