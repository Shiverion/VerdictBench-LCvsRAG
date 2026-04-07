from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def summarize(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for path in paths:
        df = load_jsonl(path)
        if df.empty:
            continue
        df["source_file"] = path.name
        frames.append(df)

    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    summary = (
        combined.groupby("condition", dropna=False)
        .apply(
            lambda x: pd.Series(
                {
                    "reviewed_n": int(len(x)),
                    "mean_legal_accuracy": float(x["legal_accuracy"].mean()),
                    "std_legal_accuracy": float(x["legal_accuracy"].std(ddof=1)) if len(x) > 1 else 0.0,
                    "mean_faithfulness": float(x["faithfulness"].mean()),
                    "corr_faithfulness_legal_accuracy": float(x[["faithfulness", "legal_accuracy"]].corr().iloc[0, 1])
                    if len(x) > 1
                    else float("nan"),
                }
            )
        )
        .reset_index()
        .sort_values("mean_legal_accuracy", ascending=False)
    )
    return combined, summary


def to_markdown(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "No human legal-accuracy files found."
    table = summary.copy()
    numeric_cols = [
        "mean_legal_accuracy",
        "std_legal_accuracy",
        "mean_faithfulness",
        "corr_faithfulness_legal_accuracy",
    ]
    for col in numeric_cols:
        table[col] = table[col].map(lambda x: round(float(x), 3) if pd.notna(x) else "")
    return table.to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more *_human.jsonl files",
    )
    parser.add_argument(
        "--out",
        default="results/legal_accuracy_summary.md",
        help="Markdown summary output path",
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.inputs]
    combined, summary = summarize(paths)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    markdown = [
        "# Human Legal Accuracy Summary",
        "",
        "## Condition-Level Summary",
        "",
        to_markdown(summary),
        "",
    ]

    if not combined.empty:
        overall_corr = combined[["faithfulness", "legal_accuracy"]].corr().iloc[0, 1]
        markdown.extend(
            [
                f"Overall correlation between automated faithfulness and human legal accuracy: **{overall_corr:.3f}**",
                "",
            ]
        )

    out_path.write_text("\n".join(markdown), encoding="utf-8")
    print(f"Wrote summary -> {out_path}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
