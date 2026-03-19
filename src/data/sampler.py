"""
Stratified 50-verdict sample selection.

Stratification axes:
  1. Document length: short / medium / long (15 / 20 / 15)
  2. Jenis perkara:   PUU / PHPU / SKLN (proportional within strata)

Output: data/metadata/sample_50.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)

STRATA_TARGETS = cfg.sampling.strata   # {"short": 15, "medium": 20, "long": 15}

# Jenis perkara priority labels for cleaner column
JENIS_LABELS = {
    "Pengujian Undang-Undang (PUU)":                           "PUU",
    "Perselisihan Hasil Pemilihan Umum (PHPU)":                "PHPU",
    "Sengketa Kewenangan Lembaga Negara (SKLN)":               "SKLN",
    "Pembubaran Partai Politik":                               "Pembubaran",
}


def select_sample(
    metadata_path: Path | None = None,
    out_path: Path | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Select a stratified 50-verdict sample from verdicts_metadata.csv.

    Selection criteria:
      - Exclude verdicts with amar_mismatch (if column present)
      - Exclude very short verdicts (< 10,000 chars) — likely incomplete extractions
      - Stratified by length stratum, then proportionally by jenis_perkara

    Returns:
        DataFrame of the 50 selected verdicts.
    """
    metadata_path = metadata_path or cfg.paths.verdicts_meta
    out_path      = out_path      or cfg.paths.sample_50

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"verdicts_metadata.csv not found at {metadata_path}. "
            "Run src.data.metadata_enricher first."
        )

    df = pd.read_csv(metadata_path)
    log.info(f"Loaded {len(df)} verdicts from {metadata_path}")

    # Normalize jenis label
    df["jenis_label"] = df["jenis_perkara"].map(JENIS_LABELS).fillna("Lainnya")

    # Quality filter
    clean = df[df["n_chars"] >= 10_000].copy()

    # Exclude amar mismatches if audit has been run
    if "amar_mismatch" in clean.columns:
        before = len(clean)
        clean = clean[clean["amar_mismatch"] != True]
        log.info(f"Excluded {before - len(clean)} amar-mismatched verdicts")

    log.info(f"Eligible pool: {len(clean)} verdicts")
    log.info(f"Stratum pool:\n{clean.stratum.value_counts().to_string()}")

    # Stratified sampling
    sampled_parts = []
    for stratum, target in STRATA_TARGETS.items():
        pool = clean[clean["stratum"] == stratum]
        if len(pool) < target:
            log.warning(
                f"Stratum '{stratum}': only {len(pool)} available, target was {target}. "
                "Using all available."
            )
            target = len(pool)

        # Within stratum: proportional by jenis_perkara
        sample = pool.groupby("jenis_label", group_keys=False).apply(
            lambda g: g.sample(
                n=max(1, round(target * len(g) / len(pool))),
                random_state=random_state,
            )
        )

        # Trim or top-up to exact target
        if len(sample) > target:
            sample = sample.sample(n=target, random_state=random_state)
        elif len(sample) < target:
            remaining = pool[~pool.index.isin(sample.index)]
            top_up = remaining.sample(
                n=min(target - len(sample), len(remaining)),
                random_state=random_state,
            )
            sample = pd.concat([sample, top_up])

        sampled_parts.append(sample)

    result = pd.concat(sampled_parts).reset_index(drop=True)
    result["sample_rank"] = range(1, len(result) + 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    log.info(f"Sample saved → {out_path}")
    log.info(f"\nFinal sample ({len(result)} verdicts):")
    log.info(f"By stratum:\n{result.stratum.value_counts().to_string()}")
    if "jenis_label" in result.columns:
        log.info(f"By jenis:\n{result.jenis_label.value_counts().to_string()}")
    else:
        log.info(f"By jenis_perkara:\n{result.jenis_perkara.value_counts().to_string()}")

    return result


if __name__ == "__main__":
    select_sample()
