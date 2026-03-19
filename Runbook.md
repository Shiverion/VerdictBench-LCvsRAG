# Runbook: LC vs RAG on MK Verdicts

**Comparative Study of Long Context vs RAG Architectures on Indonesian Constitutional Court Verdicts**

---

## Prerequisites

Before starting, make sure you have:

| Requirement | Version | Notes |
|---|---|---|
| Python | ≥ 3.10 | `python --version` |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh | sh` (preferred) |
| pip | — | `pip install --upgrade pip` (fallback) |
| Google API Key | — | [aistudio.google.com](https://aistudio.google.com) |
| OpenAI API Key | — | [platform.openai.com](https://platform.openai.com) |
| LangExtract outputs | 887 verdicts | `.txt` + `.json` pairs |
| Disk space | ~2 GB | for embeddings and results |

---

## Stage 0 — Setup

### 0.1 Extract and install

```bash
# Using uv (fastest)
uv sync

# fallback with pip
pip install -r requirements.txt
```

> **Note:** `torch` and `sentence-transformers` are large. First install takes 5–10 minutes.
> If you only want to run the data pipeline first (no experiments), install the lightweight subset:
> ```bash
> uv add python-dotenv pydantic pyyaml tqdm pandas rich google-generativeai
> ```

### 0.2 Configure API keys

```bash
# Linux / macOS
cp .env.example .env

# Windows (Command Prompt)
copy .env.example .env

# Windows (PowerShell)
Copy-Item .env.example .env
```

Open `.env` and fill in:

```bash
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

> **Important:** Never commit `.env`. It is already in `.gitignore`.

### 0.3 Copy raw LangExtract data

```bash
# Linux / macOS
cp /path/to/langextract/output/*.txt data/raw/txt/
cp /path/to/langextract/output/*.json data/raw/json/

# Windows (Command Prompt)
copy C:\path\to\output\*.txt data\raw\txt\
copy C:\path\to\output\*.json data\raw\json\

# Windows (PowerShell)
Copy-Item C:/path/to/output/*.txt data/raw/txt/
Copy-Item C:/path/to/output/*.json data/raw/json/
```

Verify:

```bash
# Linux / macOS / Windows (Git Bash)
ls data/raw/txt/ | wc -l
ls data/raw/json/ | wc -l

# Windows (PowerShell)
(Get-ChildItem data/raw/txt/).Count
(Get-ChildItem data/raw/json/).Count
```

---

## Stage 1 — Corpus Pipeline

Run the full data preparation pipeline in one command:

```bash
# Linux / macOS / Windows (Git Bash)
bash scripts/build_corpus.sh

# Windows (PowerShell - manual execution)
uv run python -m src.data.audit
uv run python -m src.data.cleaner
uv run python -m src.data.section_extractor
uv run python -m src.data.metadata_enricher
uv run python -m src.data.sampler
```

This runs 5 steps sequentially:

| Step | Script | Output |
|---|---|---|
| 1 | `src/data/audit.py` | `data/metadata/corpus_stats.csv` |
| 2 | `src/data/cleaner.py` | `data/processed/cleaned/*.txt` |
| 3 | `src/data/section_extractor.py` | `data/processed/sectioned/*.json` |
| 4 | `src/data/metadata_enricher.py` | `data/metadata/verdicts_metadata.csv` |
| 5 | `src/data/sampler.py` | `data/metadata/sample_50.csv` |

Expected runtime: **5–10 minutes** for 887 verdicts.

### 1.1 Inspect audit results (optional but recommended)

Open the first notebook to validate the corpus before proceeding:

```bash
uv run jupyter notebook notebooks/00_corpus_audit.ipynb
```

What to check in this notebook:

- **Cell 1–2:** Total verdict count, file size distribution
- **Cell 3:** Amar mismatches table — review `data/metadata/amar_mismatches.csv`. If > 50 mismatches, investigate before continuing.
- **Cell 4:** Stratum distribution chart (short/medium/long). Target: roughly 15/20/15.
- **Cell 5:** Jenis perkara breakdown. Confirm PUU, PHPU, SKLN are all represented.
- **Cell 6:** Null field heatmap — shows which metadata fields LangExtract left empty.

> **If audit reveals problems:** Re-run `uv run python -m src.data.audit` after fixing raw files. The `--skip-audit` flag on `build_corpus.sh` lets you skip re-audit on subsequent runs.

### 1.2 Validate the 50-verdict sample

```bash
uv run jupyter notebook notebooks/02_sample_selection.ipynb
```

Verify:
- Stratum counts match targets: short=15, medium=20, long=15
- No verdict appears twice
- All `file_id` values exist in `data/processed/cleaned/`

---

## Stage 2 — Build Embedding Index

Embed and index all 50 sampled verdicts using FAISS:

```bash
# Linux / macOS / Windows (Git Bash)
uv run bash scripts/build_index.sh

# Windows (PowerShell)
uv run python -m src.indexing.vector_store --all
```

Expected runtime: **10–20 minutes** (887 API calls to Google embedding endpoint, batched 100 at a time).

To use a different chunk size (e.g. for sensitivity experiments):

```bash
uv run bash scripts/build_index.sh --chunk-size 256
uv run bash scripts/build_index.sh --chunk-size 1024
```

Output: `data/processed/embedded/<verdict_id>.faiss` and `<verdict_id>.meta.pkl` for each of the 50 verdicts.

> **Cost:** Google text-embedding-004 is free within generous limits. No cost expected for 50 verdicts.

---

## Stage 3 — QA Dataset Construction

This is the most time-intensive stage. It requires human review and cannot be fully automated.

### 3.1 Generate LLM-assisted drafts

```bash
# Linux / macOS / Windows (Git Bash)
uv run bash scripts/generate_qa_drafts.sh

# Windows (PowerShell)
uv run python -m src.qa.generator
```

This calls Gemini 2.5 Flash to propose candidate questions across 4 types for each of the 50 verdicts (~350 total drafts). Runtime: **15–30 minutes**.

Output: `data/qa_dataset/qa_drafts_raw.jsonl`

> **Anti-circularity:** Draft generation uses the same Gemini model as Phase 1 generation, but faithfulness evaluation uses GPT-4o as judge. The QA draft model and the judge model are different — this prevents self-evaluation bias.

### 3.2 Human review (primary annotator)

```bash
uv run python -m src.qa.reviewer_cli
```

For each draft QA pair, you will see:

```
── 1/350 ──────────────────────────────────
Verdict:  1_PUU-XIX_2021
Type:     multi_section_reasoning
ID:       a3f9c021

┌─ Question ──────────────────────────────┐
│ Apa alasan Mahkamah menolak permohonan? │
└─────────────────────────────────────────┘

┌─ Gold Answer ───────────────────────────┐
│ Mahkamah menolak karena...              │
└─────────────────────────────────────────┘

Action [a/m/r/q]:
```

Commands:
- `a` — **Accept** as-is
- `m` — **Modify** question or answer (prompts for edits)
- `r` — **Reject** (excluded from final dataset)
- `q` — **Quit** (progress is saved, resume with `--resume`)

To resume a paused session:

```bash
uv run python -m src.qa.reviewer_cli --resume
```

Target: ~350 accepted/modified pairs. Expect **2–4 hours** of review.

Output: `data/qa_dataset/qa_pairs_full.jsonl`

### 3.3 Inter-annotator agreement check (10% sample)

Send 10% of the drafts to a second reviewer (colleague, supervisor, or legal domain expert). They run the same CLI with a different output path:

```bash
uv run python -m src.qa.reviewer_cli --resume
# Save their output as: data/qa_dataset/qa_pairs_annotator2.jsonl
```

Then compute Cohen's Kappa:

```bash
uv run python -m src.qa.iaa_calculator --annotator2 data/qa_dataset/qa_pairs_annotator2.jsonl
```

**Target: κ ≥ 0.75.** If below target, review the annotation guidelines and re-check borderline cases.

### 3.4 Build subset files

After `qa_pairs_full.jsonl` is complete, create the three subset files:

```bash
# Run this via uv for cross-platform compatibility
uv run scripts/build_subsets.py 
```

> **Note:** If `scripts/build_subsets.py` doesn't exist, create it from the snippet below:

```python
import json, random, re
from pathlib import Path

full_path = Path("data/qa_dataset/qa_pairs_full.jsonl")
if not full_path.exists():
    print("Full QA pairs file not found.")
    exit()

full = [json.loads(l) for l in full_path.read_text(encoding="utf-8").splitlines() if l.strip()]
random.seed(42)

# Ablation subset: 100 pairs, stratified by question_type
from collections import defaultdict
by_type = defaultdict(list)
for p in full:
    by_type[p["question_type"]].append(p)

targets = {"factual_extractive": 30, "multi_section_reasoning": 35,
           "structural": 20, "boundary": 15}
ablation = []
for t, n in targets.items():
    ablation.extend(random.sample(by_type[t], min(n, len(by_type[t]))))

with open("data/qa_dataset/qa_pairs_ablation.jsonl", "w", encoding="utf-8") as f:
    for p in ablation:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
print(f"Ablation subset: {len(ablation)} pairs")

# NIAH subset
niah = [p for p in full if p.get("gold_paragraph_position_pct", 0) >= 0.80]
with open("data/qa_dataset/qa_pairs_niah.jsonl", "w", encoding="utf-8") as f:
    for p in niah[:30]:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
print(f"NIAH subset: {len(niah[:30])} pairs")
```

### 3.5 Validate QA dataset

```bash
uv run jupyter notebook notebooks/03_qa_dataset_audit.ipynb
```

Check:
- Cell 1: Total pairs by type (target: 30/35/20/15% distribution)
- Cell 2: Pairs per verdict (target: ~7, no verdict with < 4 or > 12)
- Cell 3: Gold paragraph length distribution (flag empty gold_paragraphs)
- Cell 4: IAA Kappa summary

---

## Stage 4 — Run Experiments

### 4.1 Phase 1 — Controlled Baseline

Runs all 3 conditions (LC / Simple RAG / Advanced RAG) against the full 350-QA dataset using Gemini 2.5 Flash.

```bash
uv run python experiments/run_phase1.py
```

To run a single condition only:

```bash
uv run python experiments/run_phase1.py --condition lc
uv run python experiments/run_phase1.py --condition simple_rag
uv run python experiments/run_phase1.py --condition advanced_rag
```

To skip faithfulness evaluation (faster, cheaper — use for sanity check first):

```bash
uv run python experiments/run_phase1.py --no-eval
```

**Expected runtime per condition:** 30–60 minutes (350 queries + GPT-4o faithfulness eval).
**Expected cost:** ~$3–6 total for Phase 1 (Gemini generation + GPT-4o judge).

Outputs:
```
results/phase1/lc/run_YYYYMMDD_HHMMSS.jsonl
results/phase1/simple_rag/run_YYYYMMDD_HHMMSS.jsonl
results/phase1/advanced_rag/run_YYYYMMDD_HHMMSS.jsonl
results/phase1/*/cost_summary_*.csv
```

### 4.2 Phase 2 — Multi-Model

Runs the 2×3 factorial (Gemini + GPT-4o, across all 3 architectures).

```bash
uv run python experiments/run_phase2.py
```

To run one model only:

```bash
uv run python experiments/run_phase2.py --model gemini_flash
uv run python experiments/run_phase2.py --model gpt4o
```

**Expected cost:** ~$10–15 (GPT-4o generation is expensive on 350 queries × 3 conditions).

### 4.3 Ablation Study

Runs 6 Advanced RAG component conditions on the 100-question subset.

```bash
uv run python experiments/run_ablation.py
```

To run a single condition:

```bash
uv run python experiments/run_ablation.py --condition plus_reranking
uv run python experiments/run_ablation.py --condition full_advanced_rag
```

**Expected runtime:** 20–30 minutes per condition.

### 4.4 Additional Experiments

**NIAH (Needle-in-a-Haystack):**

```bash
uv run python experiments/run_niah.py
```

**Section-boundary vs fixed-size chunking:**

```bash
uv run python experiments/run_chunking_comparison.py
```

**Knowledge update scenario:**

```bash
# First, place 3 hold-out verdicts in data/raw/txt/ and data/raw/json/
# Edit experiments/run_knowledge_update.py: set NEW_VERDICT_IDS = ["id1", "id2", "id3"]
uv run python experiments/run_knowledge_update.py
```

**Run everything sequentially:**

```bash
# Full run, ~$20-25, ~4-6 hours
uv run bash scripts/run_all_experiments.sh

# Phase 1 only, ~$5, ~1 hour
uv run bash scripts/run_all_experiments.sh --phase1-only

# Skip faithfulness (fast sanity check, ~$0.50)
uv run bash scripts/run_all_experiments.sh --no-eval
```

---

## Stage 5 — Add BERTScore (optional, post-hoc)

BERTScore with IndoBERT is computationally expensive. Run it separately after experiments complete:

```bash
uv run python - << 'EOF'
from pathlib import Path
from experiments.pipeline.batch_evaluator import add_bertscore

# Add to all Phase 1 results
for jsonl in Path("results/phase1").rglob("run_*.jsonl"):
    add_bertscore(jsonl)
    print(f"Done: {jsonl}")
EOF
```

---

## Stage 6 — Human Legal Accuracy Spot-Check

Review 10% of generated answers for legal correctness (0/1/2 scale):

```bash
# Phase 1 LC condition (most recent run)
uv run python -m src.evaluation.legal_accuracy_cli \
  --results "results/phase1/lc/run_*.jsonl"

# Phase 1 Simple RAG
uv run python -m src.evaluation.legal_accuracy_cli \
  --results "results/phase1/simple_rag/run_*.jsonl"

# Phase 1 Advanced RAG
uv run python -m src.evaluation.legal_accuracy_cli \
  --results "results/phase1/advanced_rag/run_*.jsonl"
```

Each session reviews ~35 responses (10% of 350). Allow **30–45 minutes** per condition.

---

## Stage 7 — Analysis Notebooks

Run notebooks in order. Each notebook depends on outputs from prior stages.

### Notebook 00 — Corpus Audit
```bash
uv run jupyter notebook notebooks/00_corpus_audit.ipynb
```
**Requires:** `data/metadata/corpus_stats.csv`
**Produces:** Corpus size distribution plots, amar mismatch table

---

### Notebook 01 — Data Quality
```bash
uv run jupyter notebook notebooks/01_data_quality.ipynb
```
**Requires:** `corpus_stats.csv`, `amar_mismatches.csv`
**Produces:** Null field heatmap, cleaning artifact examples, data quality report

---

### Notebook 02 — Sample Selection
```bash
uv run jupyter notebook notebooks/02_sample_selection.ipynb
```
**Requires:** `sample_50.csv`, `verdicts_metadata.csv`
**Produces:** Stratum balance chart, jenis perkara distribution, sample validation table

---

### Notebook 03 — QA Dataset Audit
```bash
uv run jupyter notebook notebooks/03_qa_dataset_audit.ipynb
```
**Requires:** `qa_pairs_full.jsonl`
**Produces:** Question type distribution, pairs per verdict histogram, IAA kappa summary

---

### Notebook 04 — Phase 1 Results ⭐ (main result)
```bash
uv run jupyter notebook notebooks/04_phase1_results.ipynb
```
**Requires:** `results/phase1/*/run_*.jsonl`
**Produces:**
- Table 1: Mean faithfulness / hallucination rate / BERTScore by condition
- Table 2: Context precision and recall (RAG conditions)
- Table 3: Input token cost and latency comparison
- Figure 1: Metric comparison bar chart (LC vs Simple RAG vs Advanced RAG)
- Figure 2: Per-question-type performance breakdown
- Statistical tests: Wilcoxon signed-rank with Bonferroni correction

---

### Notebook 05 — Phase 2 Results
```bash
uv run jupyter notebook notebooks/05_phase2_results.ipynb
```
**Requires:** `results/phase2/*/run_*.jsonl`
**Produces:**
- Table 4: 2×3 factorial results (model × architecture)
- Figure 3: Interaction plot — does context window size reduce retrieval dependency?
- H₁ test: Two-way ANOVA on faithfulness, interaction effect

---

### Notebook 06 — Ablation Analysis
```bash
uv run jupyter notebook notebooks/06_ablation_analysis.ipynb
```
**Requires:** `results/ablation/*/run_*.jsonl`
**Produces:**
- Figure 4: Waterfall chart — incremental faithfulness gain per component
- Table 5: Per-component attribution (Δ faithfulness vs Simple RAG baseline)
- Figure 5: Per-component context precision lift

---

### Notebook 07 — Length Sensitivity
```bash
uv run jupyter notebook notebooks/07_length_sensitivity.ipynb
```
**Requires:** Phase 1 results + `sample_50.csv` (stratum labels)
**Produces:**
- Figure 6: Faithfulness vs document length stratum (short/medium/long) per condition
- Figure 7: Input token cost by stratum — shows LC cost explosion on long docs
- Kruskal-Wallis test: Are stratum differences significant?

---

### Notebook 08 — NIAH Analysis
```bash
uv run jupyter notebook notebooks/08_niah_analysis.ipynb
```
**Requires:** `results/additional/niah/*/run_*.jsonl`
**Produces:**
- Table 6: NIAH accuracy vs full-corpus accuracy per condition
- Figure 8: NIAH accuracy by needle depth (shallow/deep/bottom)
- Key finding: Does LC suffer "lost in the middle" on MK verdicts?

---

### Notebook 09 — Cost Frontier
```bash
uv run jupyter notebook notebooks/09_cost_frontier.ipynb
```
**Requires:** All Phase 1 + Phase 2 results
**Produces:**
- Figure 9: Pareto scatter plot (input token cost × faithfulness) — all conditions
- Figure 10: Cost per 1,000 queries projection by architecture
- Decision matrix: When is each architecture cost-optimal?

---

### Notebook 10 — Statistical Tests
```bash
uv run jupyter notebook notebooks/10_statistical_tests.ipynb
```
**Requires:** All Phase 1 results
**Produces:**
- Full Wilcoxon signed-rank test table (all pairwise comparisons, Bonferroni corrected)
- Effect sizes (Cohen's d) per metric per comparison
- Bootstrap 95% CI for all mean metrics
- Correlation: automated faithfulness vs human legal accuracy

---

## Quick Reference

### Most common commands

```bash
# Rebuild corpus from scratch
uv run bash scripts/build_corpus.sh

# Run just Phase 1 (most important experiment)
uv run python experiments/run_phase1.py

# Run Phase 1 without faithfulness eval (fast sanity check)
uv run python experiments/run_phase1.py --no-eval

# Open main results notebook
uv run jupyter notebook notebooks/04_phase1_results.ipynb

# Check test suite
uv run pytest tests/ -v

# Run tests for a single module
uv run pytest tests/evaluation/ -v
uv run pytest tests/systems/test_simple_rag.py -v
```

### Cost quick reference

| Action | Model | Est. Cost |
|---|---|---|
| Embed 50 verdicts | text-embedding-004 | Free |
| Phase 1 generation (350q × 3 cond) | Gemini 2.5 Flash | ~$0.30 |
| Phase 1 faithfulness eval | GPT-4o | ~$3–5 |
| Phase 2 generation (350q × 3 cond × GPT-4o) | GPT-4o | ~$10 |
| Ablation (100q × 6 cond) | Gemini + GPT-4o | ~$1.50 |
| **Full pipeline** | | **~$15–25** |

### Result file format

Every experiment writes JSONL — one record per QA pair:

```json
{
  "question_id": "a3f9c021",
  "verdict_id": "1_PUU-XIX_2021",
  "condition": "simple_rag_cs512_k5_fixed",
  "model": "gemini-2.5-flash-preview-04-17",
  "question": "Apa amar putusan?",
  "answer": "Mahkamah menolak permohonan Pemohon untuk seluruhnya.",
  "gold_answer": "Menolak permohonan Pemohon untuk seluruhnya.",
  "question_type": "boundary",
  "stratum": "medium",
  "input_tokens": 3241,
  "cost_usd": 0.000486,
  "latency_s": 2.14,
  "faithfulness": 0.9167,
  "hallucination_rate": 0.0833,
  "hallucination_flag": false,
  "bertscore_f1": 0.8821,
  "context_precision": 0.8,
  "context_recall": 1.0
}
```

Load all results for analysis:

```python
import pandas as pd, json
from pathlib import Path

records = []
for p in Path("results/phase1").rglob("run_*.jsonl"):
    with open(p) as f:
        records += [json.loads(l) for l in f if l.strip()]

df = pd.DataFrame(records)
df.groupby("condition")["faithfulness"].mean()
```

---

## Troubleshooting

**`FileNotFoundError: sample_50.csv`**
→ Run `bash scripts/build_corpus.sh` first. Stage 5 of the script produces this file.

**`No .txt files found in data/raw/txt`**
→ Copy LangExtract outputs:
  - Linux/macOS: `cp /path/to/langextract/*.txt data/raw/txt/`
  - Windows: `copy C:\path\to\langextract\*.txt data\raw\txt\`

**`Embedding failed after 3 attempts`**
→ Check `GOOGLE_API_KEY` in `.env`. Verify quota at [aistudio.google.com](https://aistudio.google.com).

**`openai.AuthenticationError`**
→ Check `OPENAI_API_KEY` in `.env`. GPT-4o is used for faithfulness evaluation.

**`ModuleNotFoundError: sentence_transformers`**
→ `uv add sentence_transformers` — only needed for Advanced RAG reranker.

**Jupyter not found**
→ `uv add jupyter` or `uv add jupyterlab`

**Tests failing on import**
→ Make sure you are running from the repo root: `cd lc-vs-rag-mk-verdicts && pytest`

---

## Recommended Execution Order (summary)

```
Stage 0   Setup + API keys                          ~10 min
Stage 1   bash scripts/build_corpus.sh              ~10 min
          notebooks/00_corpus_audit.ipynb           review
Stage 2   bash scripts/build_index.sh               ~20 min
Stage 3   bash scripts/generate_qa_drafts.sh        ~30 min
          python -m src.qa.reviewer_cli             ~3 hours (human)
          python -m src.qa.iaa_calculator            verify κ ≥ 0.75
          notebooks/03_qa_dataset_audit.ipynb       review
Stage 4   python experiments/run_phase1.py          ~1–2 hours
          python experiments/run_phase2.py          ~3–4 hours
          python experiments/run_ablation.py        ~2 hours
          python experiments/run_niah.py            ~30 min
          python experiments/run_chunking_comparison.py  ~30 min
Stage 5   Add BERTScore (optional)                  ~30 min
Stage 6   Legal accuracy spot-check (human)         ~2 hours
Stage 7   notebooks/04 → 10                         analysis + write-up
```

**Total active time:** ~4–6 hours (human review) + ~8–10 hours (compute, runs unattended)