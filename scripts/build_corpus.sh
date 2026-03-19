#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# build_corpus.sh
# One-shot data pipeline: audit → clean → section extract → metadata enrich
#
# Prerequisites:
#   - data/raw/txt/ and data/raw/json/ populated from LangExtract
#   - Python environment activated with requirements.txt installed
#   - .env file with API keys (only needed for QA generation, not this script)
#
# Usage:
#   bash scripts/build_corpus.sh
#   bash scripts/build_corpus.sh --skip-audit   # if audit already done
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SKIP_AUDIT=false
for arg in "$@"; do
  case $arg in
    --skip-audit) SKIP_AUDIT=true ;;
  esac
done

echo "========================================"
echo "  LC vs RAG MK Verdicts — Corpus Build  "
echo "========================================"

# Step 1: Audit
if [ "$SKIP_AUDIT" = false ]; then
  echo ""
  echo "Step 1/5: Auditing raw corpus (887 verdicts)..."
  python -m src.data.audit
  echo "✓ Audit complete → data/metadata/corpus_stats.csv"
else
  echo "Step 1/5: Skipping audit (--skip-audit flag set)"
fi

# Step 2: Clean text
echo ""
echo "Step 2/5: Cleaning verdict texts (CRLF, hyphens, page numbers)..."
python -c "
from src.data.cleaner import clean_corpus
from src.utils.config import cfg
n = clean_corpus(cfg.paths.raw_txt, cfg.paths.cleaned)
print(f'Cleaned {n} files → {cfg.paths.cleaned}')
"
echo "✓ Text cleaning complete"

# Step 3: Extract sections
echo ""
echo "Step 3/5: Extracting [X.X] sections from cleaned verdicts..."
python -c "
from src.data.section_extractor import extract_corpus
from src.utils.config import cfg
n = extract_corpus(cfg.paths.cleaned, cfg.paths.sectioned)
print(f'Extracted sections for {n} files → {cfg.paths.sectioned}')
"
echo "✓ Section extraction complete"

# Step 4: Enrich metadata
echo ""
echo "Step 4/5: Enriching metadata (filling null fields from text)..."
python -m src.data.metadata_enricher
echo "✓ Metadata enrichment complete → data/metadata/verdicts_metadata.csv"

# Step 5: Stratified sample selection
echo ""
echo "Step 5/5: Selecting stratified 50-verdict sample..."
python -m src.data.sampler
echo "✓ Sample selected → data/metadata/sample_50.csv"

echo ""
echo "========================================"
echo "  Corpus build complete!"
echo "  Next step: bash scripts/build_index.sh"
echo "========================================"
