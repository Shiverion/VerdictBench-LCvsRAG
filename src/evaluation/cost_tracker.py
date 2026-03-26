"""
Input token cost tracking and reporting.

Cost model:
  - Only INPUT tokens are tracked as primary metric
  - Output tokens are logged as supplementary but excluded from primary cost
  - Rationale: input cost IS the architectural differentiator
    (LC pays for full doc, RAG pays for chunks only)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)


IDR_PER_USD = 16_933

@dataclass
class CostMeter:
    model: str
    batch_api: bool = False
    _calls: int = 0
    _input_tokens: int = 0
    _output_tokens: int = 0

    def record(self, input_tokens: int, output_tokens: int) -> dict:
        self._calls += 1
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens

        rates = cfg.model_pricing.get(self.model, {"input": 1.00, "output": 4.00})
        if isinstance(rates, float) or isinstance(rates, int):
            rates = {"input": float(rates), "output": float(rates) * 4.0}

        multiplier = 0.5 if self.batch_api else 1.0
        input_cost  = (input_tokens  / 1_000_000) * rates.get("input", 1.0) * multiplier
        output_cost = (output_tokens / 1_000_000) * rates.get("output", 4.0) * multiplier
        total = input_cost + output_cost

        return {
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            "cost_usd":      round(total, 6),
        }

    def summary(self) -> dict:
        rates = cfg.model_pricing.get(self.model, {"input": 1.00, "output": 4.00})
        if isinstance(rates, float) or isinstance(rates, int):
            rates = {"input": float(rates), "output": float(rates) * 4.0}

        multiplier = 0.5 if self.batch_api else 1.0
        total_input_cost  = (self._input_tokens  / 1_000_000) * rates.get("input", 1.0) * multiplier
        total_output_cost = (self._output_tokens / 1_000_000) * rates.get("output", 4.0) * multiplier
        total_usd = total_input_cost + total_output_cost

        return {
            "model":               self.model,
            "batch_api":           self.batch_api,
            "total_calls":         self._calls,
            "total_input_tokens":  self._input_tokens,
            "total_output_tokens": self._output_tokens,
            "total_cost_usd":      round(total_usd, 4),
            "total_cost_idr":      round(total_usd * IDR_PER_USD, 0),
        }

    def log_summary(self, label: str = ""):
        s = self.summary()
        prefix = f"[{label}] " if label else ""
        log.info(
            f"{prefix}Cost summary | calls={s['total_calls']} | "
            f"input={s['total_input_tokens']:,} | output={s['total_output_tokens']:,} | "
            f"${s['total_cost_usd']:.4f} | Rp{s['total_cost_idr']:,.0f}"
        )


class BudgetExceeded(Exception):
    """Raised when a cost threshold is breached during a run."""
    pass


@dataclass
class CircuitBreaker:
    """
    Monitors cumulative token/cost consumption during a run.
    Raises BudgetExceeded if any threshold is crossed.
    """
    max_input_tokens:  int | None   = None
    max_output_tokens: int | None   = None
    max_cost_usd:      float | None = None
    max_cost_idr:      float | None = None
    warn_at_pct:       float        = 0.80

    _warned: set = field(default_factory=set)

    def check(self, *meters: CostMeter, label: str = "") -> None:
        total_input  = sum(m._input_tokens  for m in meters)
        total_output = sum(m._output_tokens for m in meters)
        total_usd    = sum(m.summary()["total_cost_usd"] for m in meters)
        total_idr    = total_usd * IDR_PER_USD

        checks = {
            "input_tokens":  (total_input,  self.max_input_tokens,  f"{total_input:,} input tokens"),
            "output_tokens": (total_output, self.max_output_tokens, f"{total_output:,} output tokens"),
            "cost_usd":      (total_usd,    self.max_cost_usd,      f"${total_usd:.4f}"),
            "cost_idr":      (total_idr,    self.max_cost_idr,      f"Rp{total_idr:,.0f}"),
        }

        prefix = f"[{label}] " if label else ""

        for key, (current, limit, display) in checks.items():
            if limit is None:
                continue

            pct = current / limit
            if pct >= self.warn_at_pct and key not in self._warned:
                self._warned.add(key)
                log.warning(
                    f"{prefix}⚠️  Budget warning: {display} "
                    f"= {pct*100:.1f}% of {key} limit ({limit:,})"
                )

            if current >= limit:
                msg = (
                    f"{prefix}🛑 Circuit breaker triggered: {display} "
                    f"exceeded limit {limit:,} for '{key}'. "
                    f"Partial results saved."
                )
                log.error(msg)
                raise BudgetExceeded(msg)


def check_budget(n_pairs: int, gen_model: str, judge_model: str, budget_idr: float = 60_000):
    gen_rates = cfg.model_pricing.get(gen_model, {"input": 1.0})
    if not isinstance(gen_rates, dict): gen_rates = {"input": float(gen_rates)}
    judge_rates = cfg.model_pricing.get(judge_model, {"input": 2.0})
    if not isinstance(judge_rates, dict): judge_rates = {"input": float(judge_rates)}
    
    # Check if config has token_estimates, else default to safe P95
    est_gen_input = n_pairs * getattr(getattr(cfg, "token_estimates", None), "gen_input_p95", 100000)
    est_judge_input = n_pairs * getattr(getattr(cfg, "token_estimates", None), "judge_input_p95", 15000)
    
    est_cost = (
        (est_gen_input   / 1e6) * gen_rates.get("input", 1.0) +
        (est_judge_input / 1e6) * judge_rates.get("input", 2.0)
    )
    
    # 50% discount for Gemini batch API
    if getattr(cfg.models, 'use_batch_api', False):
        est_cost *= 0.5
        
    est_idr = est_cost * IDR_PER_USD

    if est_idr > budget_idr:
        raise RuntimeError(f"Projected cost Rp{est_idr:,.0f} exceeds budget Rp{budget_idr:,.0f}.")
    log.info(f"Pre-flight cost estimate: Rp{est_idr:,.0f} (budget: Rp{budget_idr:,.0f}) ✅")

@dataclass
class CostRecord:
    question_id:  str
    verdict_id:   str
    condition:    str
    model:        str
    input_tokens: int
    cost_usd:     float
    stratum:      str = ""   # short / medium / long — populated from sample metadata


@dataclass
class CostTracker:
    """Accumulates cost records across an experiment run."""
    records: list[CostRecord] = field(default_factory=list)

    def add(self, record: CostRecord) -> None:
        self.records.append(record)

    def summary(self) -> pd.DataFrame:
        """Per-condition cost summary."""
        if not self.records:
            return pd.DataFrame()
        df = pd.DataFrame([r.__dict__ for r in self.records])
        return (
            df.groupby(["condition", "model"])
            .agg(
                n_queries        =("question_id", "count"),
                total_tokens     =("input_tokens", "sum"),
                mean_tokens      =("input_tokens", "mean"),
                total_cost_usd   =("cost_usd", "sum"),
                mean_cost_usd    =("cost_usd", "mean"),
                cost_per_1k_queries=("cost_usd", lambda x: x.mean() * 1000),
            )
            .reset_index()
            .round(6)
        )

    def by_stratum(self) -> pd.DataFrame:
        """Cost breakdown by document length stratum."""
        if not self.records:
            return pd.DataFrame()
        df = pd.DataFrame([r.__dict__ for r in self.records])
        if "stratum" not in df.columns or df.stratum.isna().all():
            return pd.DataFrame()
        return (
            df.groupby(["condition", "stratum"])
            .agg(
                mean_tokens   =("input_tokens", "mean"),
                mean_cost_usd =("cost_usd", "mean"),
            )
            .reset_index()
            .round(6)
        )

    def save(self, path: Path) -> None:
        pd.DataFrame([r.__dict__ for r in self.records]).to_csv(path, index=False)
        log.info(f"Cost records saved → {path}")


def cost_report(results_jsonl: Path) -> pd.DataFrame:
    """
    Generate cost summary from a completed results JSONL file.
    Useful for post-hoc analysis without re-running experiments.
    """
    import json
    records = []
    with open(results_jsonl) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records.append({
                    "condition":    r.get("condition"),
                    "model":        r.get("model"),
                    "input_tokens": r.get("input_tokens", 0),
                    "cost_usd":     r.get("cost_usd", 0.0),
                    "stratum":      r.get("stratum", ""),
                })
    df = pd.DataFrame(records)
    summary = df.groupby(["condition", "model"]).agg(
        n_queries=("input_tokens", "count"),
        total_tokens=("input_tokens", "sum"),
        mean_tokens=("input_tokens", "mean"),
        total_cost_usd=("cost_usd", "sum"),
        mean_cost_usd=("cost_usd", "mean"),
    ).reset_index().round(6)
    return summary
