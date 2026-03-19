#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# generate_qa_drafts.sh
# LLM-assisted QA draft generation for all 50 sampled verdicts.
# Output: data/qa_dataset/qa_drafts_raw.jsonl (for human review)
#
# Requires: GOOGLE_API_KEY set in .env
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

echo "========================================"
echo "  Generating QA Drafts (LLM-assisted)"
echo "========================================"
echo "  WARNING: Uses non-evaluator model to avoid circularity."
echo "  Generated drafts must be reviewed by human annotator."
echo ""

python -c "
from src.qa.generator import generate_all_drafts
from src.utils.config import cfg
generate_all_drafts(
    sample_csv=cfg.paths.sample_50,
    cleaned_dir=cfg.paths.cleaned,
    out_path=cfg.paths.qa_dir / 'qa_drafts_raw.jsonl',
    qa_per_verdict=cfg.sampling.qa_per_verdict,
)
"

echo ""
echo "✓ QA drafts generated → data/qa_dataset/qa_drafts_raw.jsonl"
echo ""
echo "Next step: python -m src.qa.reviewer_cli"
echo "           (human review: accept / modify / reject each draft)"
