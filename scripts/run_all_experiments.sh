#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_all_experiments.sh
# Sequential execution of all experiment scripts.
# Estimated cost: ~$15–25 for full run with Gemini 2.5 Flash + GPT-4o judge.
#
# Usage:
#   bash scripts/run_all_experiments.sh
#   bash scripts/run_all_experiments.sh --no-eval    # skip faithfulness (cheap pass)
#   bash scripts/run_all_experiments.sh --phase1-only
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

NO_EVAL=""
PHASE1_ONLY=false

for arg in "$@"; do
  case $arg in
    --no-eval)     NO_EVAL="--no-eval" ;;
    --phase1-only) PHASE1_ONLY=true ;;
  esac
done

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

echo "========================================"
echo "  LC vs RAG — Full Experiment Suite"
echo "  Started: $(timestamp)"
echo "========================================"

echo ""
echo "─── Phase 1: Controlled Baseline ───────"
python experiments/run_phase1.py $NO_EVAL
echo "✓ Phase 1 complete at $(timestamp)"

if [ "$PHASE1_ONLY" = false ]; then
  echo ""
  echo "─── Phase 2: Multi-Model ────────────────"
  python experiments/run_phase2.py $NO_EVAL
  echo "✓ Phase 2 complete at $(timestamp)"

  echo ""
  echo "─── Ablation Study ──────────────────────"
  python experiments/run_ablation.py $NO_EVAL
  echo "✓ Ablation complete at $(timestamp)"

  echo ""
  echo "─── Additional: NIAH ────────────────────"
  python experiments/run_niah.py $NO_EVAL
  echo "✓ NIAH complete at $(timestamp)"

  echo ""
  echo "─── Additional: Chunking Comparison ─────"
  python experiments/run_chunking_comparison.py $NO_EVAL
  echo "✓ Chunking comparison complete at $(timestamp)"

  echo ""
  echo "─── Additional: Knowledge Update ────────"
  python experiments/run_knowledge_update.py $NO_EVAL
  echo "✓ Knowledge update complete at $(timestamp)"
fi

echo ""
echo "========================================"
echo "  All experiments complete!"
echo "  Finished: $(timestamp)"
echo "  Results → results/"
echo "  Next: open notebooks/04_phase1_results.ipynb"
echo "========================================"
