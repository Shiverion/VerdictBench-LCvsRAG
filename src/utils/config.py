"""
Central configuration — all hyperparameters, paths, and model settings in one place.
Load via: from src.utils.config import cfg
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(find_dotenv() or str(ROOT / ".env"), override=True)


# ── Model pricing ($ per 1M input tokens) ─────────────────────────────────────
MODEL_PRICING: dict[str, float] = {
    "gemini-2.5-flash-preview-04-17": 0.15,
    "gpt-4o":                          2.50,
    "gpt-4o-mini":                     0.15,
    "claude-opus-4.6":                 15.00,  # Elite tier pricing
    "gpt-5.4":                        10.00,  # next-gen pricing
    "gemini-3.1-pro":                  1.25,   # pro tier pricing
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
    phase2_model_b:  str = Field(default_factory=lambda: os.getenv("PHASE2_MODEL_B", "gpt-4o-mini"))
    judge_model:     str = "gemini-3-flash-preview"
    judge_thinking_level: str = "none"
    use_batch_api:   bool = True
    embedding_model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2-preview"))
    claude_reviewer: str = Field(default_factory=lambda: os.getenv("CLAUDE_REVIEWER", "claude-opus-4.6"))
    gpt_reviewer:    str = Field(default_factory=lambda: os.getenv("GPT_REVIEWER", "gpt-5.4"))
    gemini_reviewer: str = Field(default_factory=lambda: os.getenv("GEMINI_REVIEWER", "gemini-3.1-pro"))
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


class TokenEstimates(BaseModel):
    gen_input_p95: int = 15_000
    judge_input_p95: int = 1_500


class BudgetConfig(BaseModel):
    max_input_tokens:  int = 15_000_000
    max_output_tokens: int = 750_000
    max_cost_usd:      float = 3.50
    max_cost_idr:      float = 55_000
    warn_at_pct:       float = 0.80


class Config(BaseModel):
    paths:    PathsConfig    = PathsConfig()
    models:   ModelConfig    = ModelConfig()
    rag:      RAGConfig      = RAGConfig()
    eval:     EvalConfig     = EvalConfig()
    sampling: SamplingConfig = SamplingConfig()
    token_estimates: TokenEstimates = TokenEstimates()
    budget:   BudgetConfig   = BudgetConfig()
    keys:     dict[str, str | None] = {
        "google_api_key":    os.getenv("GOOGLE_API_KEY"),
        "openai_api_key":    os.getenv("OPENAI_API_KEY"),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
    }
    model_pricing: dict[str, float] = MODEL_PRICING

    class Config:
        arbitrary_types_allowed = True


cfg = Config()
