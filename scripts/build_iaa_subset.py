from __future__ import annotations

import json
import random
from pathlib import Path


SOURCE = Path("data/qa_dataset/qa_pairs_full.jsonl")
TARGET = Path("data/qa_dataset/qa_pairs_iaa_30.jsonl")
N_ITEMS = 30
SEED = 42


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"Missing source dataset: {SOURCE}")

    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < N_ITEMS:
        raise SystemExit(f"Need at least {N_ITEMS} rows in {SOURCE}, found {len(rows)}")

    rng = random.Random(SEED)
    sample = rng.sample(rows, N_ITEMS)
    TARGET.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in sample) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(sample)} shared IAA items to {TARGET}")


if __name__ == "__main__":
    main()
