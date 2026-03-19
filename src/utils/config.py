"""
Central configuration — all hyperparameters, paths, and model settings in one place.
Load via: from src.utils.config import cfg
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]


# ── Model pricing ($ per 1M input tokens) ─────────────────────────────────────
MODEL_PRICING: dict[str, float] = {
    "gemini-2.5-flash-preview-04-17": 0.15,
    "gpt-4o":                          2.50,
    "gpt-4o-mini":                     0.15,
}


class PathsConfig(BaseModel):
    root:            Path = ROOT
    raw_txt:         Path = ROOT / "data/raw/txt"
    raw_json:        Path = ROOT / "data/raw/json"
    cleaned:         Path = ROOT / "data/processed/cleaned"
    sectioned:       Path = ROOT / "data/processed/sectioned"
    embedded:        Path = ROOT / "data/processed/embedded"
    metadata_dir:    Path = ROOT / "data/metadata"
    qa_dir:          Path = ROOT / "data/qa_dataset"
    qa_drafts:       Path = ROOT / "data/qa_dataset/qa_drafts_raw.jsonl"
    qa_pairs:        Path = ROOT / "data/qa_dataset/qa_pairs_full.jsonl"
    results:         Path = ROOT / "results"
    corpus_stats:    Path = ROOT / "data/metadata/corpus_stats.csv"
    verdicts_meta:   Path = ROOT / "data/metadata/verdicts_metadata.csv"
    amar_mismatches: Path = ROOT / "data/metadata/amar_mismatches.csv"
    sample_50:       Path = ROOT / "data/metadata/sample_50.csv"

    class Config:
        arbitrary_types_allowed = True


class ModelConfig(BaseModel):
    phase1_model:    str = Field(default_factory=lambda: os.getenv("PHASE1_MODEL", "gemini-2.5-flash-preview-04-17"))
    phase2_model_a:  str = Field(default_factory=lambda: os.getenv("PHASE2_MODEL_A", "gemini-2.5-flash-preview-04-17"))
    phase2_model_b:  str = Field(default_factory=lambda: os.getenv("PHASE2_MODEL_B", "gpt-4o"))
    judge_model:     str = "gpt-4o"        # evaluator — always different from generation model
    embedding_model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2-preview"))
    reranker_model:  str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    temperature:     float = 0.0
    max_output_tokens: int = 512


class RAGConfig(BaseModel):
    # Tuned parameters (grid searched in sensitivity experiment)
    chunk_sizes:   list[int] = [256, 512, 1024]
    chunk_size:    int = 512          # default for Phase 1
    chunk_overlap_ratio: float = 0.10  # 10% of chunk size
    top_k_values:  list[int] = [3, 5, 10]
    top_k:         int = 5            # default for Phase 1
    rerank_top_m:  int = 3            # after reranking, keep top m ≤ top_k


class EvalConfig(BaseModel):
    hallucination_threshold: float = 0.20   # HR > 0.20 → binary flag
    jaccard_threshold:       float = 0.50   # for context precision/recall
    human_spot_check_ratio:  float = 0.10   # 10% of responses reviewed by human
    judge_spot_check_ratio:  float = 0.05   # 5% of faithfulness judgments reviewed by human
    iaa_target_kappa:        float = 0.75   # inter-annotator agreement target


class SamplingConfig(BaseModel):
    total_verdicts:    int = 50
    strata: dict[str, int] = {"short": 15, "medium": 20, "long": 15}
    # Page thresholds (converted to char count at ~2000 chars/page)
    short_max_chars:  int = 60_000    # < ~30 pages
    long_min_chars:   int = 200_000   # > ~100 pages
    qa_pairs_target:  int = 350
    ablation_subset:  int = 100
    niah_subset:      int = 30
    qa_per_verdict:   int = 7


class Config(BaseModel):
    paths:    PathsConfig    = PathsConfig()
    models:   ModelConfig    = ModelConfig()
    rag:      RAGConfig      = RAGConfig()
    eval:     EvalConfig     = EvalConfig()
    sampling: SamplingConfig = SamplingConfig()
    model_pricing: dict[str, float] = MODEL_PRICING

    class Config:
        arbitrary_types_allowed = True


cfg = Config()
