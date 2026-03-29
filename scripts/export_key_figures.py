from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from notebooks.notebook_utils import (
    load_ablation_results,
    load_phase1_results,
    load_phase2_results,
    load_sample,
    phase1_failure_by_stratum,
    set_plot_style,
)


ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures"


def save(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def export_phase2_faithfulness() -> None:
    _, summary = load_phase2_results()
    plot_df = summary.copy()
    plot_df[["ci_low", "ci_high"]] = pd.DataFrame(plot_df["faithfulness_ci"].tolist(), index=plot_df.index)
    plot_df["err_low"] = plot_df["mean_faithfulness"] - plot_df["ci_low"]
    plot_df["err_high"] = plot_df["ci_high"] - plot_df["mean_faithfulness"]
    order = ["Simple RAG", "Advanced RAG", "Long Context"]
    hue_order = ["Gemini 2.5 Flash", "GPT-4o Mini"]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=plot_df,
        x="architecture",
        y="mean_faithfulness",
        hue="model_family",
        order=order,
        hue_order=hue_order,
        palette=["#1f77b4", "#ff7f0e"],
        ax=ax,
    )
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("")
    ax.set_ylabel("Mean faithfulness")
    ax.set_title("Phase 2: Retrieval remains strongest across both model families")

    patches = ax.patches
    for i, (_, row) in enumerate(plot_df.sort_values(["architecture", "model_family"], key=lambda s: s.map({v: k for k, v in enumerate(order + hue_order)})).iterrows()):
        if i >= len(patches):
            break
        patch = patches[i]
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height()
        ax.errorbar(
            x=x,
            y=y,
            yerr=[[row["err_low"]], [row["err_high"]]],
            fmt="none",
            ecolor="black",
            capsize=4,
            linewidth=1.1,
        )

    save(fig, "phase2_faithfulness.png")


def export_phase1_failures() -> None:
    fail_df = phase1_failure_by_stratum()
    strata = ["short", "medium", "long"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=fail_df,
        x="stratum",
        y="failure_rate",
        hue="stratum",
        order=strata,
        palette=["#9ecae1", "#6baed6", "#de2d26"],
        legend=False,
        ax=ax,
    )
    ax.set_ylim(0, 0.65)
    ax.set_xlabel("")
    ax.set_ylabel("LC failure rate")
    ax.set_title("Phase 1 Long Context failures are concentrated in long verdicts")
    for patch, (_, row) in zip(ax.patches, fail_df.set_index("stratum").reindex(strata).reset_index().iterrows()):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + 0.02,
            f'{int(row["failure_rows"])}/{int(row["total_rows"])}',
            ha="center",
            va="bottom",
            fontsize=10,
        )
    save(fig, "phase1_lc_failure_by_stratum.png")


def export_ablation() -> None:
    _, summary = load_ablation_results()
    order = [
        "Baseline Simple RAG",
        "+ Query Rewrite",
        "+ Metadata Filter",
        "+ Hybrid Search",
        "+ Reranking",
        "Full Advanced RAG",
    ]
    palette = ["#9ecae1", "#fdae6b", "#74c476", "#31a354", "#e6550d", "#756bb1"]
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=summary,
        x="condition_label",
        y="mean_faithfulness",
        hue="condition_label",
        order=order,
        palette=palette,
        legend=False,
        ax=ax,
    )
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("")
    ax.set_ylabel("Mean faithfulness")
    ax.set_title("Ablation: hybrid search helps most, reranking hurts most")
    ax.tick_params(axis="x", rotation=20)
    for patch in ax.patches:
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + 0.02,
            f"{patch.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    save(fig, "ablation_faithfulness.png")


def export_cost_frontier() -> None:
    _, summary = load_phase2_results()
    palette = {
        "Simple RAG": "#2ca02c",
        "Advanced RAG": "#ff7f0e",
        "Long Context": "#d62728",
    }
    markers = {
        "Gemini 2.5 Flash": "o",
        "GPT-4o Mini": "s",
    }
    labels = {
        ("Gemini 2.5 Flash", "Simple RAG"): "G25-SR",
        ("Gemini 2.5 Flash", "Advanced RAG"): "G25-AR",
        ("Gemini 2.5 Flash", "Long Context"): "G25-LC",
        ("GPT-4o Mini", "Simple RAG"): "G4O-SR",
        ("GPT-4o Mini", "Advanced RAG"): "G4O-AR",
        ("GPT-4o Mini", "Long Context"): "G4O-LC",
    }
    offsets = {
        "G25-SR": (8, 8),
        "G25-AR": (8, -10),
        "G25-LC": (8, 8),
        "G4O-SR": (8, 8),
        "G4O-AR": (8, -10),
        "G4O-LC": (8, 8),
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in summary.iterrows():
        x = row["mean_total_cost_usd"]
        y = row["mean_faithfulness"]
        label = labels[(row["model_family"], row["architecture"])]
        ax.scatter(
            x,
            y,
            s=130,
            color=palette[row["architecture"]],
            marker=markers[row["model_family"]],
            alpha=0.9,
            edgecolor="black",
            linewidth=0.6,
        )
        dx, dy = offsets[label]
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=10)

    ax.set_xlabel("Mean cost per query (USD)")
    ax.set_ylabel("Mean faithfulness")
    ax.set_title("Cost frontier: Long Context is dominated on both quality and cost")
    ax.set_xscale("log")

    # Small in-figure key for compact labels.
    ax.text(
        0.02,
        0.04,
        "G25 = Gemini 2.5 Flash, G4O = GPT-4o Mini\nSR = Simple RAG, AR = Advanced RAG, LC = Long Context",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    save(fig, "cost_faithfulness_frontier.png")


def main() -> None:
    set_plot_style()
    export_phase2_faithfulness()
    export_phase1_failures()
    export_ablation()
    export_cost_frontier()
    print(f"Exported figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
