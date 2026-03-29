# Final To-Do

This file is now a **pre-submission paper checklist**. The major paper-facing items except the human legal validation pass have been completed.

## Remaining Submission Blocker

### 1. Human Legal Accuracy Spot-Check

**Status:** not done  
**Why it matters:** the paper still relies on an automated faithfulness judge without a reported human legal validation pass. For a legal NLP venue, this is the main remaining methodological gap.  
**Time:** about 3 to 4 hours  
**Effort:** high but straightforward

**What to do**

- Complete the human legal accuracy spot-check on a 10% sample.
- Summarize the results in the paper.
- Relate the human 0/1/2 ratings to the automated faithfulness signal.

**Recommended scope**

- Review 10% of generated answers for each Phase 1 condition.
- That is about 30 answers per condition.
- Total human review load: about 90 answers.

**Run**

```bash
uv run python -m src.evaluation.legal_accuracy_cli \
  --results "results/phase1/lc/run_*.jsonl"

uv run python -m src.evaluation.legal_accuracy_cli \
  --results "results/phase1/simple_rag/run_*.jsonl"

uv run python -m src.evaluation.legal_accuracy_cli \
  --results "results/phase1/advanced_rag/run_*.jsonl"
```

**Scoring rubric**

- `0` = factually wrong or unsupported
- `1` = partially correct
- `2` = legally accurate and grounded

**How to report it**

- Add a small table with mean legal accuracy per Phase 1 condition.
- Compare that ranking against the automated faithfulness ranking.
- If the ranking is consistent (`Simple RAG > Advanced RAG > Long Context`), state that this supports the validity of the automated judge.

**Done when**

- The report contains a short subsection or table summarizing human legal review.
- The automated judge is no longer presented without human validation.

---

## Completed in This Pass

These items were pending previously and are now done in the paper/docs:

1. **Phase 2 effect sizes.** `report.md` now includes Cohen's *d* with bootstrap confidence intervals for all main Phase 2 pairwise comparisons.
2. **Knowledge-update decision.** The `n = 3` pilot has been demoted from the main empirical claim and is treated as exploratory only.
3. **Phase 1 LC failure analysis by stratum.** The report now states that all 51 Phase 1 LC failures occur in the `long` stratum.
4. **BERTScore coverage confirmation.** The report now states that all scorable records in the main Phase 1, Phase 2, and ablation tables have populated `bertscore_f1` values.
5. **Abstract trim.** The abstract in `report.md` has been reduced and no longer leans on the underpowered knowledge-update pilot.
6. **Title unification.** `README.md`, `report.md`, and the citation title now use the same paper title.

---

## Optional Follow-Up Polish

These are not blocking after the current pass, but they would still improve the repo:

- Clean the remaining README formatting artifacts near the top section if you want a fully polished public landing page.
- Add a small human-legal-validation sentence to `README.md` after the spot-check is complete, so the repo overview matches the paper.
