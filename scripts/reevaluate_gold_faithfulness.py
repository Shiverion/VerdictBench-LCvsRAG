from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from src.evaluation.faithfulness import evaluate_faithfulness
from src.evaluation.hallucination import compute_hallucination
from src.utils.config import cfg


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_ROOT = ROOT / "results_corrected" / "gold_evidence_faithfulness"


@dataclass(frozen=True)
class ResultInput:
    collection: str
    label: str
    path: Path


DEFAULT_INPUTS = [
    ResultInput(
        "phase1",
        "simple_rag",
        RESULTS / "phase1/simple_rag/run_20260320_080919_bs.jsonl",
    ),
    ResultInput(
        "phase1",
        "advanced_rag",
        RESULTS / "phase1/advanced_rag/run_20260320_164547_bs.jsonl",
    ),
    ResultInput("phase1", "lc", RESULTS / "phase1/lc/run_20260320_073844_bs.jsonl"),
    ResultInput(
        "phase2",
        "gemini_flash_simple_rag",
        RESULTS / "phase2/gemini_flash/simple_rag/results_clean_bs.jsonl",
    ),
    ResultInput(
        "phase2",
        "gemini_flash_advanced_rag",
        RESULTS / "phase2/gemini_flash/advanced_rag/results_clean_bs.jsonl",
    ),
    ResultInput(
        "phase2",
        "gemini_flash_lc",
        RESULTS / "phase2/gemini_flash/lc/results_clean_bs.jsonl",
    ),
    ResultInput(
        "phase2",
        "gpt4o_simple_rag",
        RESULTS / "phase2/gpt4o/simple_rag/results_clean_bs.jsonl",
    ),
    ResultInput(
        "phase2",
        "gpt4o_advanced_rag",
        RESULTS / "phase2/gpt4o/advanced_rag/results_clean_bs.jsonl",
    ),
    ResultInput("phase2", "gpt4o_lc", RESULTS / "phase2/gpt4o/lc/results_clean_bs.jsonl"),
    ResultInput(
        "niah",
        "simple_rag",
        RESULTS / "additional/niah/simple_rag/run_20260326_134009_bs.jsonl",
    ),
    ResultInput(
        "niah",
        "advanced_rag",
        RESULTS / "additional/niah/advanced_rag/run_20260326_134651_bs.jsonl",
    ),
    ResultInput("niah", "lc", RESULTS / "additional/niah/lc/run_20260326_133347_bs.jsonl"),
    ResultInput(
        "ablation",
        "simple_rag_baseline",
        RESULTS / "ablation/simple_rag_baseline/results_clean_bs.jsonl",
    ),
    ResultInput(
        "ablation",
        "plus_query_rewrite",
        RESULTS / "ablation/plus_query_rewrite/results_clean_bs.jsonl",
    ),
    ResultInput(
        "ablation",
        "plus_metadata_filter",
        RESULTS / "ablation/plus_metadata_filter/results_clean_bs.jsonl",
    ),
    ResultInput(
        "ablation",
        "plus_hybrid_search",
        RESULTS / "ablation/plus_hybrid_search/results_clean_bs.jsonl",
    ),
    ResultInput(
        "ablation",
        "plus_reranking",
        RESULTS / "ablation/plus_reranking/results_clean_bs.jsonl",
    ),
    ResultInput(
        "ablation",
        "full_advanced_rag",
        RESULTS / "ablation/full_advanced_rag/results_clean_bs.jsonl",
    ),
]


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def select_inputs(collections: set[str], explicit_inputs: list[str] | None) -> list[ResultInput]:
    if explicit_inputs:
        selected: list[ResultInput] = []
        default_by_path = {item.path.resolve(): item for item in DEFAULT_INPUTS}
        for input_path in explicit_inputs:
            resolved = Path(input_path).resolve()
            selected.append(
                default_by_path.get(
                    resolved,
                    ResultInput("custom", resolved.stem, resolved),
                )
            )
        return selected
    return [item for item in DEFAULT_INPUTS if item.collection in collections]


def output_path_for(input_path: Path, limit_per_file: int | None) -> Path:
    input_path = input_path.resolve()
    try:
        rel = input_path.relative_to(RESULTS)
    except ValueError:
        rel = Path("custom") / input_path.name

    stem = input_path.stem
    suffix = f"_sample{limit_per_file}_goldfaith" if limit_per_file else "_goldfaith"
    return OUT_ROOT / rel.with_name(stem + suffix + input_path.suffix)


def validate_records(path: Path, records: list[dict]) -> dict:
    missing_gold_for_answered = []
    missing_answer_for_scored = []
    unscored = []
    skipped_no_answer = []

    for idx, record in enumerate(records, start=1):
        has_answer = bool((record.get("answer") or "").strip())
        is_scored = record.get("faithfulness") is not None
        has_gold = bool(record.get("gold_paragraphs"))

        if has_answer and is_scored and not has_gold:
            missing_gold_for_answered.append((idx, record.get("question_id")))
        if is_scored and not has_answer:
            missing_answer_for_scored.append((idx, record.get("question_id")))
        if not is_scored:
            unscored.append((idx, record.get("question_id")))
        if not has_answer:
            skipped_no_answer.append((idx, record.get("question_id")))

    return {
        "path": str(path),
        "records": len(records),
        "missing_gold_for_answered_scored": missing_gold_for_answered,
        "missing_answer_for_scored": missing_answer_for_scored,
        "unscored_original_records": unscored,
        "skipped_no_answer": skipped_no_answer,
    }


def eligible_records(records: list[dict]) -> list[dict]:
    return [
        record
        for record in records
        if record.get("gold_paragraphs")
        and (record.get("answer") or "").strip()
        and record.get("faithfulness") is not None
    ]


def choose_records(records: list[dict], limit_per_file: int | None, seed: int) -> list[dict]:
    selected = eligible_records(records)
    if limit_per_file is None or len(selected) <= limit_per_file:
        return selected
    rng = random.Random(seed)
    return rng.sample(selected, limit_per_file)


def add_gold_evidence_faithfulness(record: dict, judge_model: str | None) -> dict:
    gold_context = "\n\n".join(record.get("gold_paragraphs") or [])
    faith = evaluate_faithfulness(
        answer=record.get("answer", ""),
        context=gold_context,
        judge_model=judge_model,
    )
    usage = faith.pop("usage", {"input_tokens": 0, "output_tokens": 0})
    hallucination = compute_hallucination(faith["faithfulness"])

    output = dict(record)
    output.update(
        {
            "gold_evidence_faithfulness": faith["faithfulness"],
            "gold_evidence_n_statements": faith["n_statements"],
            "gold_evidence_n_supported": faith["n_supported"],
            "gold_evidence_statement_details": faith["statement_details"],
            "gold_evidence_hallucination_rate": hallucination["hallucination_rate"],
            "gold_evidence_hallucination_flag": hallucination["hallucination_flag"],
            "gold_evidence_context_chars": len(gold_context),
            "gold_evidence_context_paragraphs": len(record.get("gold_paragraphs") or []),
            "gold_evidence_judge_model": judge_model or cfg.models.judge_model,
            "gold_evidence_judge_input_tokens": usage.get("input_tokens", 0),
            "gold_evidence_judge_output_tokens": usage.get("output_tokens", 0),
            "gold_evidence_evaluated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return output


def evaluate_records(
    records: list[dict],
    judge_model: str | None,
    workers: int,
    desc: str,
) -> list[dict]:
    workers = max(1, workers)
    if workers == 1:
        return [
            add_gold_evidence_faithfulness(record, judge_model)
            for record in tqdm(records, desc=desc)
        ]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(
            tqdm(
                executor.map(
                    lambda record: add_gold_evidence_faithfulness(record, judge_model),
                    records,
                ),
                total=len(records),
                desc=desc,
            )
        )


def print_validation(item: ResultInput, validation: dict, selected_count: int, out_path: Path) -> None:
    print(f"\n[{item.collection}] {item.label}")
    print(f"  input: {item.path}")
    print(f"  records: {validation['records']}")
    print(f"  eligible: {selected_count}")
    print(f"  output: {out_path}")
    for key in [
        "missing_gold_for_answered_scored",
        "missing_answer_for_scored",
        "unscored_original_records",
        "skipped_no_answer",
    ]:
        values = validation[key]
        print(f"  {key}: {len(values)}")
        if values:
            preview = ", ".join(f"line {line}:{qid}" for line, qid in values[:5])
            print(f"    first: {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate saved answers against gold_paragraphs."
    )
    parser.add_argument(
        "--collections",
        nargs="+",
        choices=["phase1", "phase2", "niah", "ablation", "all"],
        default=["phase1", "phase2", "niah", "ablation"],
        help="Default result collections to process.",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="Explicit JSONL input paths. Overrides --collections.",
    )
    parser.add_argument(
        "--limit-per-file",
        type=int,
        default=None,
        help="Randomly evaluate at most N eligible records per input file.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel records to evaluate. Keep conservative to avoid rate limits.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually call the judge and write outputs. Omit for validation-only dry run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing corrected output files.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip corrected output files that already exist.",
    )
    args = parser.parse_args()

    collections = set(args.collections)
    if "all" in collections:
        collections = {"phase1", "phase2", "niah", "ablation"}

    inputs = select_inputs(collections, args.inputs)
    if not inputs:
        raise SystemExit("No input files selected.")

    print("Gold-evidence faithfulness re-evaluation")
    print(f"mode: {'RUN' if args.run else 'VALIDATION ONLY'}")
    print(f"judge_model: {args.judge_model or cfg.models.judge_model}")
    print(f"limit_per_file: {args.limit_per_file}")
    print(f"workers: {args.workers}")
    print(f"output_root: {OUT_ROOT}")

    any_validation_errors = False
    for item in inputs:
        if not item.path.exists():
            raise SystemExit(f"Missing input file: {item.path}")

        records = load_jsonl(item.path)
        validation = validate_records(item.path, records)
        selected = choose_records(records, args.limit_per_file, args.seed)
        out_path = output_path_for(item.path, args.limit_per_file)
        print_validation(item, validation, len(selected), out_path)

        blocking_missing = (
            validation["missing_gold_for_answered_scored"]
            or validation["missing_answer_for_scored"]
        )
        if blocking_missing:
            any_validation_errors = True

        if args.run:
            if blocking_missing:
                raise SystemExit(f"Blocking validation errors for {item.path}")
            if out_path.exists() and not args.overwrite:
                if args.skip_existing:
                    print(f"  skipped_existing: {out_path}")
                    continue
                raise SystemExit(f"Output exists; use --overwrite: {out_path}")

            corrected = evaluate_records(
                records=selected,
                judge_model=args.judge_model,
                workers=args.workers,
                desc=f"{item.collection}:{item.label}",
            )
            write_jsonl(out_path, corrected)
            print(f"  wrote: {out_path}")

    if any_validation_errors and not args.run:
        raise SystemExit("Validation found blocking missing gold/answer fields.")

    if not args.run:
        print("\nValidation complete. No API calls were made and no outputs were written.")


if __name__ == "__main__":
    main()
