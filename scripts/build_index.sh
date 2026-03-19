#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# build_index.sh
# Build FAISS embeddings and BM25 indexes for all 50 sampled verdicts.
# Requires GOOGLE_API_KEY to be set (embedding API calls).
#
# Usage:
#   bash scripts/build_index.sh
#   bash scripts/build_index.sh --chunk-size 256   # override chunk size
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

CHUNK_SIZE=512
for arg in "$@"; do
  case $arg in
    --chunk-size=*) CHUNK_SIZE="${arg#*=}" ;;
  esac
done

echo "========================================"
echo "  Building FAISS Index (chunk=$CHUNK_SIZE)"
echo "========================================"

python -c "
import pandas as pd
from pathlib import Path
from src.utils.config import cfg
from src.indexing.chunkers.fixed_size import FixedSizeChunker
from src.indexing.chunkers.section_boundary import SectionBoundaryChunker
from src.indexing.vector_store import VerdictIndexRegistry
from src.data.section_extractor import VerdictSections
from src.utils.logger import get_logger
import json

log = get_logger('build_index')

sample = pd.read_csv(cfg.paths.sample_50)
registry = VerdictIndexRegistry()
chunk_size = $CHUNK_SIZE

log.info(f'Building indexes for {len(sample)} verdicts, chunk_size={chunk_size}')

for _, row in sample.iterrows():
    vid = row['file_id']
    if registry.exists(vid):
        log.info(f'  Skipping {vid} (already indexed)')
        continue

    txt_path = cfg.paths.cleaned / f'{vid}.txt'
    if not txt_path.exists():
        log.warning(f'  Missing cleaned text: {vid}')
        continue

    text   = txt_path.read_text(encoding='utf-8')
    chunks = FixedSizeChunker(chunk_size=chunk_size).split(text, verdict_id=vid)
    registry.build_and_save(vid, chunks)
    log.info(f'  Indexed {vid}: {len(chunks)} chunks')

log.info('All indexes built.')
"

echo ""
echo "✓ FAISS index build complete → data/processed/embedded/"
echo ""
echo "Next step: bash scripts/generate_qa_drafts.sh"
