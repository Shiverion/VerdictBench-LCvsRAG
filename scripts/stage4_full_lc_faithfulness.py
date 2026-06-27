"""
Stage 4: Full LC context faithfulness re-evaluation (50-record stratified pilot).

Reads from Stage 2 goldfaith outputs (which already carry gold_evidence_faithfulness)
and adds full_lc_faithfulness by reconstructing the actual LC generation context
from cleaned verdict texts. Context is capped at 100k chars (vs original 8k bug cap).

Stratification (total = 50):
  - Phase 1 LC long stratum: 20 records (random from 39 eligible)
  - Phase 2 Gemini Flash LC: 12 records (random)
  - Phase 2 GPT-4o LC: 12 records (random, windowed context)
  - NIAH LC: 6 records (random)

Outputs to results_corrected/full_lc_faithfulness/
Each output file has both gold_evidence_faithfulness (from Stage 2) and
full_lc_faithfulness (computed here) for direct paired comparison.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from src.evaluation.faithfulness import evaluate_faithfulness
from src.evaluation.hallucination import compute_hallucination
from src.utils.config import cfg


GPT4O_MAX_CONTEXT_CHARS = 100_000 * 4
GPT4O_WINDOW_HALF = 50_000 * 4


def _find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (
            (parent / "pyproject.toml").exists()
            and (parent / "results").exists()
            and (parent / "results_corrected").exists()
        ):
            return parent
    raise RuntimeError(
        "Cannot locate project root (need pyproject.toml, results/, and results_corrected/)"
    )


ROOT = _find_project_root()
GOLDFAITH_ROOT = ROOT / "results_corrected" / "gold_evidence_faithfulness"
CLEANED = ROOT / "data" / "processed" / "cleaned"
OUT_ROOT = ROOT / "results_corrected" / "full_lc_faithfulness"


@dataclass(frozen=True)
class LCInput:
    label: str
    goldfaith_path: Path
    windowed: bool


LC_INPUTS = [
    LCInput(
        "phase1_lc",
        GOLDFAITH_ROOT / "phase1/lc/run_20260320_073844_bs_goldfaith.jsonl",
        windowed=False,
    ),
    LCInput(
        "phase2_gemini_flash_lc",
        GOLDFAITH_ROOT / "phase2/gemini_flash/lc/results_clean_bs_goldfaith.jsonl",
        windowed=False,
    ),
    LCInput(
        "phase2_gpt4o_lc",
        GOLDFAITH_ROOT / "phase2/gpt4o/lc/results_clean_bs_goldfaith.jsonl",
        windowed=True,
    ),
    LCInput(
        "niah_lc",
        GOLDFAITH_ROOT / "additional/niah/lc/run_20260326_133347_bs_goldfaith.jsonl",
        windowed=False,
    ),
]

STRATA_SPEC: dict[str, dict] = {
    "phase1_lc":              {"stratum_filter": "long", "n": 20},
    "phase2_gemini_flash_lc": {"stratum_filter": None,   "n": 12},
    "phase2_gpt4o_lc":        {"stratum_filter": None,   "n": 12},
    "niah_lc":                {"stratum_filter": None,   "n": 6},
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def eligible(record: dict) -> bool:
    return bool(
        (record.get("answer") or "").strip()
        and record.get("verdict_id")
        and record.get("faithfulness") is not None
        and record.get("gold_evidence_faithfulness") is not None
    )


def stratified_sample(inp: LCInput, seed: int) -> list[dict]:
    records = load_jsonl(inp.goldfaith_path)
    spec = STRATA_SPEC[inp.label]
    pool = [r for r in records if eligible(r)]

    if spec["stratum_filter"]:
        pool = [r for r in pool if r.get("stratum") == spec["stratum_filter"]]

    n = spec["n"]
    if n is None or len(pool) <= n:
        return pool
    rng = random.Random(seed)
    return rng.sample(pool, n)


def reconstruct_lc_context(verdict_id: str, windowed: bool) -> str:
    txt_path = CLEANED / f"{verdict_id}.txt"
    if not txt_path.exists():
        raise FileNotFoundError(f"Cleaned verdict not found: {txt_path}")
    text = txt_path.read_text(encoding="utf-8")
    if windowed and len(text) > GPT4O_MAX_CONTEXT_CHARS:
        first = text[:GPT4O_WINDOW_HALF]
        last = text[-GPT4O_WINDOW_HALF:]
        text = (
            first
            + "\n\n[... TENGAH DOKUMEN DIPOTONG UNTUK KETERBATASAN KONTEKS ...]\n\n"
            + last
        )
    return text


def add_full_lc_faithfulness(record: dict, windowed: bool, judge_model: str | None) -> dict:
    context = reconstruct_lc_context(record["verdict_id"], windowed)
    faith = evaluate_faithfulness(
        answer=record.get("answer", ""),
        context=context,
        judge_model=judge_model,
        max_context_chars=100_000,
    )
    usage = faith.pop("usage", {"input_tokens": 0, "output_tokens": 0})
    hallucination = compute_hallucination(faith["faithfulness"])

    output = dict(record)
    output.update(
        {
            "full_lc_faithfulness": faith["faithfulness"],
            "full_lc_n_statements": faith["n_statements"],
            "full_lc_n_supported": faith["n_supported"],
            "full_lc_statement_details": faith["statement_details"],
            "full_lc_hallucination_rate": hallucination["hallucination_rate"],
            "full_lc_hallucination_flag": hallucination["hallucination_flag"],
            "full_lc_context_chars": len(context),
            "full_lc_windowed": windowed,
            "full_lc_judge_model": judge_model or cfg.models.judge_model,
            "full_lc_judge_input_tokens": usage.get("input_tokens", 0),
            "full_lc_judge_output_tokens": usage.get("output_tokens", 0),
            "full_lc_evaluated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return output


def evaluate_records(
    records: list[dict],
    windowed: bool,
    judge_model: str | None,
    workers: int,
    desc: str,
) -> list[dict]:
    workers = max(1, workers)
    fn = lambda r: add_full_lc_faithfulness(r, windowed, judge_model)
    if workers == 1:
        return [fn(r) for r in tqdm(records, desc=desc)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(tqdm(ex.map(fn, records), total=len(records), desc=desc))


def print_summary(label: str, corrected: list[dict]) -> None:
    full_vals = [r["full_lc_faithfulness"] for r in corrected]
    gold_vals = [r["gold_evidence_faithfulness"] for r in corrected]
    deltas = [f - g for f, g in zip(full_vals, gold_vals)]
    n = len(corrected)
    print(f"  n: {n}")
    print(f"  full_lc mean:         {sum(full_vals)/n:.4f}")
    print(f"  gold_evidence mean:   {sum(gold_vals)/n:.4f}")
    print(f"  delta mean (full-gold): {sum(deltas)/n:+.4f}")
    print(f"  delta range: [{min(deltas):+.4f}, {max(deltas):+.4f}]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4: Evaluate LC answers against full generation context."
    )
    parser.add_argument("--judge-model", default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers. Keep low (1-2) to avoid rate limits.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run", action="store_true", help="Make API calls and write output.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    print("Stage 4: Full LC context faithfulness (50-record pilot, 100k char cap)")
    print(f"mode:        {'RUN' if args.run else 'VALIDATION ONLY'}")
    print(f"judge_model: {args.judge_model or cfg.models.judge_model}")
    print(f"output_root: {OUT_ROOT}")
    print(f"project_root:{ROOT}")

    all_selected = 0
    for inp in LC_INPUTS:
        if not inp.goldfaith_path.exists():
            raise SystemExit(f"Missing Stage 2 goldfaith file: {inp.goldfaith_path}")

        selected = stratified_sample(inp, args.seed)
        all_selected += len(selected)
        out_path = OUT_ROOT / f"{inp.label}_sample50_100k_fullctxfaith.jsonl"

        print(f"\n[{inp.label}]")
        print(f"  goldfaith input: {inp.goldfaith_path}")
        print(f"  selected:  {len(selected)}")
        print(f"  windowed:  {inp.windowed}")
        print(f"  output:    {out_path}")
        if inp.label == "phase1_lc":
            print(f"  strata:    {dict(Counter(r.get('stratum') for r in selected))}")

        if args.run:
            if out_path.exists() and not args.overwrite:
                if args.skip_existing:
                    print(f"  skipped_existing: {out_path}")
                    continue
                raise SystemExit(f"Output exists; use --overwrite or --skip-existing: {out_path}")

            corrected = evaluate_records(
                records=selected,
                windowed=inp.windowed,
                judge_model=args.judge_model,
                workers=args.workers,
                desc=inp.label,
            )
            write_jsonl(out_path, corrected)
            print(f"  wrote: {out_path}")
            print_summary(inp.label, corrected)

    print(f"\nTotal records selected: {all_selected}")
    if not args.run:
        print("\nValidation complete. No API calls made, no files written.")


if __name__ == "__main__":
    main()
