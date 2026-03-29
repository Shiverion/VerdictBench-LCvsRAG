from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


COMMON_IMPORTS = """from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from notebooks.notebook_utils import *

set_plot_style()
ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
"""


def write_notebook(path: Path, cells: list) -> None:
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = cells
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3.13"}
    path.write_text(nbf.writes(notebook), encoding="utf-8")


def notebook_map() -> dict[str, list]:
    return {
        "00_corpus_audit.ipynb": [
            md("# 00. Corpus Audit\n\nThis notebook validates the corpus audit stage described in Runbook Stage 1 and Stage 7."),
            code(COMMON_IMPORTS + "\ncorpus = load_corpus_stats()\nmismatches = load_amar_mismatches()\nprint(f'Total verdicts audited: {len(corpus)}')\ncorpus.head()"),
            md("## Corpus size distribution"),
            code("corpus[['n_chars', 'est_pages']].describe().round(2)"),
            code("fig, axes = plt.subplots(1, 2, figsize=(14, 5))\nsns.histplot(corpus['n_chars'], bins=30, ax=axes[0], color='#1f77b4')\naxes[0].set_title('Character count distribution')\nsns.histplot(corpus['est_pages'], bins=30, ax=axes[1], color='#ff7f0e')\naxes[1].set_title('Estimated pages distribution')\nplt.tight_layout()"),
            md("## Amar mismatch audit"),
            code("print(f'Amar mismatches: {len(mismatches)}')\nmismatches[['file_id', 'nomor_putusan', 'amar_json', 'amar_text']].head(10)"),
            md("## Representation checks"),
            code("corpus['jenis_perkara'].value_counts().rename_axis('jenis_perkara').to_frame('count')"),
            code("null_summary = corpus[['tanggal_putusan', 'amar_text', 'amar_json']].isna().mean().mul(100).round(2)\nnull_summary.to_frame('percent_null')"),
        ],
        "01_data_quality.ipynb": [
            md("# 01. Data Quality\n\nThis notebook inspects audit anomalies, missing metadata, and cleaning-sensitive fields."),
            code(COMMON_IMPORTS + "\ncorpus = load_corpus_stats()\nmeta = load_verdict_metadata()\nmismatches = load_amar_mismatches()\nprint(corpus.shape, meta.shape, mismatches.shape)"),
            md("## Missingness profile"),
            code("quality = pd.DataFrame({\n    'corpus_stats_missing_pct': corpus.isna().mean().mul(100),\n    'verdict_metadata_missing_pct': meta.isna().mean().mul(100),\n}).fillna(0).sort_values('verdict_metadata_missing_pct', ascending=False)\nquality.head(20).round(2)"),
            code("heatmap_data = quality.head(12)\nsns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd')\nplt.title('Top missingness rates (%)')\nplt.tight_layout()"),
            md("## Amar mismatch examples"),
            code("mismatches[['file_id', 'nomor_putusan', 'amar_json', 'amar_text']].head(15)"),
            md("## Potential cleaning artifacts"),
            code("artifact_flags = corpus.assign(\n    no_amar_section=~corpus['has_amar_section'].astype(bool),\n    missing_panel=~corpus['has_panel_ketua'].astype(bool),\n    missing_date=~corpus['has_tanggal_putusan'].astype(bool),\n)\nartifact_flags[['no_amar_section', 'missing_panel', 'missing_date']].mean().mul(100).round(2).to_frame('percent_of_corpus')"),
        ],
        "02_sample_selection.ipynb": [
            md("# 02. Sample Selection\n\nThis notebook validates the stratified 50-verdict benchmark sample."),
            code(COMMON_IMPORTS + "\nsample = load_sample()\nmeta = load_verdict_metadata()\nmerged = sample.merge(meta[['file_id', 'jenis_perkara', 'est_pages']], on='file_id', how='left', suffixes=('', '_meta'))\nprint(f'Sample size: {len(sample)}')\nmerged.head()"),
            md("## Stratum balance"),
            code("sample['stratum'].value_counts().rename_axis('stratum').to_frame('count')"),
            code("sns.countplot(data=sample, x='stratum', order=['short', 'medium', 'long'], palette='crest')\nplt.title('Benchmark sample by length stratum')\nplt.tight_layout()"),
            md("## Duplicate and existence checks"),
            code("pd.DataFrame({\n    'duplicate_file_ids': [int(sample['file_id'].duplicated().sum())],\n    'missing_metadata_rows': [int(merged['jenis_perkara'].isna().sum())],\n})"),
            md("## Case-type coverage"),
            code("merged['jenis_perkara'].fillna('unknown').value_counts().rename_axis('jenis_perkara').to_frame('count')"),
            code("sns.boxplot(data=sample, x='stratum', y='est_pages', order=['short', 'medium', 'long'], palette='Set2')\nplt.title('Estimated pages by selected stratum')\nplt.tight_layout()"),
        ],
        "03_qa_dataset_audit.ipynb": [
            md("# 03. QA Dataset Audit\n\nThis notebook audits the final 300-pair QA dataset and links it to the overlap IAA result."),
            code(COMMON_IMPORTS + "\nqa = load_qa_pairs()\nsample = load_sample()[['file_id', 'stratum']].rename(columns={'file_id': 'verdict_id'})\nqa = qa.merge(sample, on='verdict_id', how='left')\nprint(f'Accepted QA pairs: {len(qa)}')\nqa.head()"),
            md("## Question type balance"),
            code("qa['question_type'].value_counts().rename_axis('question_type').to_frame('count')"),
            code("sns.countplot(data=qa, y='question_type', order=qa['question_type'].value_counts().index, palette='viridis')\nplt.title('Question type distribution')\nplt.tight_layout()"),
            md("## Pairs per verdict"),
            code("pairs_per_verdict = qa['verdict_id'].value_counts().sort_values(ascending=False)\npairs_per_verdict.describe().round(2).to_frame('value')"),
            code("sns.histplot(pairs_per_verdict, bins=15, color='#2ca02c')\nplt.title('Pairs per verdict')\nplt.xlabel('QA pairs')\nplt.tight_layout()"),
            md("## Gold evidence integrity"),
            code("pd.DataFrame({\n    'empty_gold_paragraph_lists': [int((qa['n_gold_paragraphs'] == 0).sum())],\n    'mean_gold_paragraphs_per_pair': [round(float(qa['n_gold_paragraphs'].mean()), 2)],\n    'unique_verdicts': [int(qa['verdict_id'].nunique())],\n})"),
            md("## Inter-annotator agreement"),
            code("iaa_path = ROOT / 'data/qa_dataset/iaa_existing_overlap/kappa_summary.json'\niaa = pd.read_json(iaa_path, typ='series')\niaa"),
        ],
        "04_phase1_results.ipynb": [
            md("# 04. Phase 1 Results\n\nMain architecture comparison for Gemini 2.5 Flash on the 300-question benchmark."),
            code(COMMON_IMPORTS + "\nphase1_raw, phase1_summary = load_phase1_results()\nphase1_scored = phase1_raw[phase1_raw['faithfulness'].notna()].copy()\nmarkdown_df(phase1_summary)"),
            md("## Table 1-style summary"),
            code("table1 = markdown_df(phase1_summary[['architecture', 'queries', 'mean_faithfulness', 'mean_bertscore_f1', 'mean_hallucination_rate', 'zero_faithfulness_cases', 'mean_input_tokens', 'total_cost_usd', 'total_latency_s']])\ntable1"),
            md("## Retrieval metrics for RAG conditions"),
            code("rag_only = phase1_summary[phase1_summary['architecture'].isin(['Simple RAG', 'Advanced RAG'])][['architecture', 'mean_context_precision', 'mean_context_recall']]\nmarkdown_df(rag_only)"),
            md("## Cost and latency comparison"),
            code("cost_table = phase1_summary[['architecture', 'mean_input_tokens', 'mean_total_cost_usd', 'total_cost_usd', 'total_latency_s']]\nmarkdown_df(cost_table)"),
            code("fig, axes = plt.subplots(1, 3, figsize=(18, 5))\nsns.barplot(data=phase1_summary, x='architecture', y='mean_faithfulness', ax=axes[0], palette='crest')\naxes[0].set_title('Faithfulness')\nsns.barplot(data=phase1_summary, x='architecture', y='mean_hallucination_rate', ax=axes[1], palette='flare')\naxes[1].set_title('Hallucination rate')\nsns.barplot(data=phase1_summary, x='architecture', y='mean_bertscore_f1', ax=axes[2], palette='mako')\naxes[2].set_title('BERTScore F1')\nfor ax in axes:\n    ax.tick_params(axis='x', rotation=20)\nplt.tight_layout()"),
            md("## By question type"),
            code("question_breakdown = phase1_scored.groupby(['architecture', 'question_type'])['faithfulness'].mean().reset_index()\nquestion_breakdown.pivot(index='question_type', columns='architecture', values='faithfulness').round(3)"),
            code("sns.barplot(data=question_breakdown, x='question_type', y='faithfulness', hue='architecture')\nplt.xticks(rotation=20)\nplt.title('Phase 1 faithfulness by question type')\nplt.tight_layout()"),
            md("## Statistical tests and LC failure concentration"),
            code("phase1_tests = phase1_pairwise_tests()\nmarkdown_df(phase1_tests)"),
            code("failure_stratum = phase1_failure_by_stratum()\nmarkdown_df(failure_stratum)"),
        ],
        "05_phase2_results.ipynb": [
            md("# 05. Phase 2 Results\n\nTwo-model by three-architecture comparison on complete 300-query runs."),
            code(COMMON_IMPORTS + "\nphase2, phase2_summary = load_phase2_results()\nphase2_effects = phase2_effect_sizes()\nphase2_summary[['ci_low', 'ci_high']] = pd.DataFrame(phase2_summary['faithfulness_ci'].tolist(), index=phase2_summary.index)\nmarkdown_df(phase2_summary[['model_family', 'architecture', 'queries', 'mean_faithfulness', 'ci_low', 'ci_high', 'mean_bertscore_f1', 'mean_hallucination_rate']])"),
            md("## Factorial summary"),
            code("phase2_pivot = phase2_summary.pivot(index='architecture', columns='model_family', values='mean_faithfulness').round(3)\nphase2_pivot"),
            code("sns.pointplot(data=phase2_summary, x='architecture', y='mean_faithfulness', hue='model_family', markers='o')\nplt.title('Interaction plot: model family x architecture')\nplt.tight_layout()"),
            md("## Two-way ANOVA on per-question faithfulness"),
            code("import statsmodels.api as sm\nfrom statsmodels.formula.api import ols\nanova_model = ols('faithfulness ~ C(model_family) * C(architecture)', data=phase2).fit()\nsm.stats.anova_lm(anova_model, typ=2).round(4)"),
            md("## Effect sizes"),
            code("markdown_df(phase2_effects[['model_family', 'comparison', 'mean_difference', 'cohens_d', 'wilcoxon_p']])"),
            code("phase2['architecture_order'] = pd.Categorical(phase2['architecture'], categories=['Simple RAG', 'Advanced RAG', 'Long Context'], ordered=True)\nsns.boxplot(data=phase2, x='architecture_order', y='faithfulness', hue='model_family')\nplt.xlabel('Architecture')\nplt.title('Per-question faithfulness distributions')\nplt.tight_layout()"),
        ],
        "06_ablation_analysis.ipynb": [
            md("# 06. Ablation Analysis\n\nComponent-level analysis of the Advanced RAG pipeline."),
            code(COMMON_IMPORTS + "\nablation, ablation_summary = load_ablation_results()\nmarkdown_df(ablation_summary[['condition_label', 'queries', 'mean_faithfulness', 'mean_bertscore_f1', 'mean_hallucination_rate', 'mean_context_precision', 'mean_total_cost_usd']])"),
            md("## Faithfulness deltas versus baseline"),
            code("baseline = float(ablation_summary.loc[ablation_summary['condition_label'] == 'Baseline Simple RAG', 'mean_faithfulness'].iloc[0])\ndeltas = ablation_summary[['condition_label', 'mean_faithfulness']].copy()\ndeltas['delta_vs_baseline'] = deltas['mean_faithfulness'] - baseline\nmarkdown_df(deltas)"),
            code("sns.barplot(data=deltas, y='condition_label', x='delta_vs_baseline', palette='coolwarm')\nplt.axvline(0, color='black', linewidth=1)\nplt.title('Faithfulness delta vs Simple RAG baseline')\nplt.tight_layout()"),
            md("## Incremental metric view"),
            code("metric_cols = ['mean_faithfulness', 'mean_hallucination_rate', 'mean_context_precision']\nablation_summary.set_index('condition_label')[metric_cols].round(3)"),
            code("sns.lineplot(data=ablation_summary, x='condition_label', y='mean_faithfulness', marker='o')\nplt.xticks(rotation=30, ha='right')\nplt.title('Faithfulness across ablation stages')\nplt.tight_layout()"),
        ],
        "07_length_sensitivity.ipynb": [
            md("# 07. Length Sensitivity\n\nThis notebook stratifies Phase 1 performance by short, medium, and long verdicts."),
            code(COMMON_IMPORTS + "\nphase1_raw, _ = load_phase1_results()\nphase1_scored = phase1_raw[phase1_raw['faithfulness'].notna()].copy()\nphase1_scored['effective_input_tokens'] = phase1_scored['gen_input_tokens'].fillna(phase1_scored['input_tokens'])\nsample = load_sample()[['file_id', 'stratum']].rename(columns={'file_id': 'verdict_id'})\nphase1_scored = phase1_scored.merge(sample, on='verdict_id', how='left', suffixes=('', '_sample'))\nphase1_scored['stratum_final'] = phase1_scored['stratum_sample'].fillna(phase1_scored['stratum'])\nphase1_scored[['architecture', 'verdict_id', 'stratum_final', 'faithfulness']].head()"),
            md("## Faithfulness by stratum"),
            code("length_summary = phase1_scored.groupby(['architecture', 'stratum_final'])[['faithfulness', 'effective_input_tokens', 'total_cost_usd']].mean().reset_index()\nmarkdown_df(length_summary)"),
            code("sns.barplot(data=length_summary, x='stratum_final', y='faithfulness', hue='architecture', order=['short', 'medium', 'long'])\nplt.title('Phase 1 faithfulness by document-length stratum')\nplt.tight_layout()"),
            md("## Cost explosion by stratum"),
            code("sns.barplot(data=length_summary, x='stratum_final', y='effective_input_tokens', hue='architecture', order=['short', 'medium', 'long'])\nplt.title('Mean input tokens by stratum')\nplt.tight_layout()"),
            md("## Kruskal-Wallis tests within architecture"),
            code("rows = []\nfor architecture, group in phase1_scored.groupby('architecture'):\n    values = [bucket['faithfulness'].to_numpy() for _, bucket in group.groupby('stratum_final') if len(bucket) > 0]\n    if len(values) >= 2:\n        stat, p = stats.kruskal(*values)\n        rows.append({'architecture': architecture, 'kruskal_stat': stat, 'pvalue': p})\nmarkdown_df(pd.DataFrame(rows))"),
        ],
        "08_niah_analysis.ipynb": [
            md("# 08. Needle-in-a-Haystack Analysis\n\nThis notebook evaluates deep-evidence questions and approximates needle depth by locating gold paragraphs inside the cleaned verdict text."),
            code(COMMON_IMPORTS + "\nsummary = pd.read_csv(ROOT / 'results/additional/niah/niah_summary.csv')\nniah, niah_summary = load_niah_results()\ndepth_summary = niah_by_depth()\nsummary"),
            md("## NIAH condition summary"),
            code("markdown_df(niah_summary[['architecture', 'queries', 'mean_faithfulness', 'mean_bertscore_f1', 'zero_faithfulness_cases']])"),
            code("summary_plot = summary.assign(condition=summary['condition'].replace({'simple_rag_cs512_k5_fixed':'Simple RAG', 'advanced_rag_QR_MF_HS_RR':'Advanced RAG', 'lc':'Long Context'}))\nsns.barplot(data=summary_plot, x='condition', y='niah_accuracy', palette='crest')\nplt.title('NIAH accuracy by architecture')\nplt.tight_layout()"),
            md("## Approximate depth buckets"),
            code("markdown_df(depth_summary)"),
            code("depth_plot = depth_summary[depth_summary['depth_bucket'] != 'unknown']\nif not depth_plot.empty:\n    sns.barplot(data=depth_plot, x='depth_bucket', y='mean_faithfulness', hue='architecture', order=['shallow', 'deep', 'bottom'])\n    plt.title('NIAH faithfulness by approximate needle depth')\n    plt.tight_layout()\nelse:\n    print('Depth buckets could not be reconstructed from the available cleaned texts.')"),
        ],
        "09_cost_frontier.ipynb": [
            md("# 09. Cost Frontier\n\nThis notebook maps faithfulness against cost across the main experiment conditions."),
            code(COMMON_IMPORTS + "\nphase1_raw, phase1_summary = load_phase1_results()\nphase2, phase2_summary = load_phase2_results()\nphase1_summary['label'] = 'Phase 1 | ' + phase1_summary['architecture']\nphase2_summary['label'] = phase2_summary['model_family'] + ' | ' + phase2_summary['architecture']\ncombined = pd.concat([\n    phase1_summary[['label', 'mean_faithfulness', 'mean_total_cost_usd', 'total_cost_usd', 'mean_input_tokens']].assign(experiment='Phase 1'),\n    phase2_summary[['label', 'mean_faithfulness', 'mean_total_cost_usd', 'total_cost_usd', 'mean_input_tokens']].assign(experiment='Phase 2'),\n], ignore_index=True)\nshort_map = {\n    'Phase 1 | Simple RAG': 'P1-SR',\n    'Phase 1 | Advanced RAG': 'P1-AR',\n    'Phase 1 | Long Context': 'P1-LC',\n    'Gemini 2.5 Flash | Simple RAG': 'G25-SR',\n    'Gemini 2.5 Flash | Advanced RAG': 'G25-AR',\n    'Gemini 2.5 Flash | Long Context': 'G25-LC',\n    'GPT-4o Mini | Simple RAG': 'G4O-SR',\n    'GPT-4o Mini | Advanced RAG': 'G4O-AR',\n    'GPT-4o Mini | Long Context': 'G4O-LC',\n}\ncombined['short_label'] = combined['label'].map(short_map)\nmarkdown_df(combined[['short_label', 'label', 'mean_faithfulness', 'mean_total_cost_usd']].sort_values('mean_faithfulness', ascending=False))"),
            md("## Pareto-style scatter"),
            code("plt.figure(figsize=(12, 7))\nsns.scatterplot(data=combined, x='mean_total_cost_usd', y='mean_faithfulness', hue='experiment', style='experiment', s=150)\noffsets = [(8, 8), (8, -12), (-10, 8), (-10, -12), (12, 0), (-14, 0), (0, 12), (0, -16), (14, 12)]\nfor idx, (_, row) in enumerate(combined.iterrows()):\n    dx, dy = offsets[idx % len(offsets)]\n    plt.annotate(row['short_label'],\n                 (row['mean_total_cost_usd'], row['mean_faithfulness']),\n                 textcoords='offset points',\n                 xytext=(dx, dy),\n                 ha='center',\n                 va='center',\n                 fontsize=9,\n                 fontweight='bold',\n                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.75))\nplt.xscale('log')\nplt.title('Faithfulness vs mean cost per query')\nplt.grid(True, which='both', ls='--', alpha=0.5)\nplt.tight_layout()"),
            md("## Projected cost per 1,000 queries"),
            code("combined['cost_per_1000_queries'] = combined['mean_total_cost_usd'] * 1000\nmarkdown_df(combined[['label', 'cost_per_1000_queries', 'mean_faithfulness']].sort_values('cost_per_1000_queries'))"),
        ],
        "10_statistical_tests.ipynb": [
            md("# 10. Statistical Tests\n\nThis notebook centralizes inferential analysis for the benchmark. The human legal-accuracy correlation section remains gated until those scores are collected."),
            code(COMMON_IMPORTS + "\nphase1_tests = phase1_pairwise_tests()\nphase2_effects = phase2_effect_sizes()\ncoverage = bertscore_coverage()\nphase1_tests"),
            md("## Phase 1 pairwise Wilcoxon tests"),
            code("markdown_df(phase1_tests)"),
            md("## Phase 2 effect sizes and Wilcoxon p-values"),
            code("markdown_df(phase2_effects[['model_family', 'comparison', 'mean_difference', 'cohens_d', 'wilcoxon_p']])"),
            md("## Bootstrap confidence intervals for main mean metrics"),
            code("phase2, phase2_summary = load_phase2_results()\nintervals = phase2_summary[['model_family', 'architecture', 'mean_faithfulness', 'faithfulness_ci']].copy()\nintervals[['ci_low', 'ci_high']] = pd.DataFrame(intervals['faithfulness_ci'].tolist(), index=intervals.index)\nmarkdown_df(intervals[['model_family', 'architecture', 'mean_faithfulness', 'ci_low', 'ci_high']])"),
            md("## BERTScore coverage check"),
            code("markdown_df(coverage[['collection', 'model_family', 'architecture', 'condition_label', 'scorable_rows', 'bertscore_rows', 'coverage_rate']].fillna(''))"),
            md("## Human legal accuracy correlation (pending)\n\nRun this section after completing the manual 0/1/2 legal-accuracy spot-check and writing those values back into the result files."),
            code("phase1_raw, _ = load_phase1_results()\nlegal_rows = phase1_raw[phase1_raw['legal_accuracy'].notna()].copy()\nif legal_rows.empty:\n    print('No human legal-accuracy labels found yet. Re-run after the spot-check is complete.')\nelse:\n    print(legal_rows[['architecture', 'faithfulness', 'legal_accuracy']].corr(numeric_only=True))"),
        ],
    }


def build_notebooks() -> None:
    for name, cells in notebook_map().items():
        write_notebook(NOTEBOOKS / name, cells)


if __name__ == "__main__":
    build_notebooks()
