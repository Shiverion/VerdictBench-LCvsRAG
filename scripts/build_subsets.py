import json, random, re
from pathlib import Path

# Config
full_path = Path("data/qa_dataset/qa_pairs_full.jsonl")
ablation_path = Path("data/qa_dataset/qa_pairs_ablation.jsonl")
niah_path = Path("data/qa_dataset/qa_pairs_niah.jsonl")

if not full_path.exists():
    print(f"Error: {full_path} not found. Run review CLI first.")
    exit(1)

# Load full dataset
lines = full_path.read_text(encoding="utf-8").splitlines()
full = [json.loads(l) for l in lines if l.strip()]
random.seed(42)

print(f"Loaded {len(full)} QA pairs.")

# 1. Ablation subset: 100 pairs, stratified by question_type
from collections import defaultdict
by_type = defaultdict(list)
for p in full:
    by_type[p["question_type"]].append(p)

targets = {
    "factual_extractive": 30, 
    "multi_section_reasoning": 35,
    "structural": 20, 
    "boundary": 15
}

ablation = []
for t, n in targets.items():
    available = by_type[t]
    sample_size = min(n, len(available))
    ablation.extend(random.sample(available, sample_size))

with ablation_path.open("w", encoding="utf-8") as f:
    for p in ablation:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
print(f"Created {ablation_path} ({len(ablation)} pairs)")

# 2. NIAH subset: top 30 pairs with highest gold_paragraph_position_pct
# (fallback to first 30 if field missing)
niah = sorted(full, key=lambda x: x.get("gold_paragraph_position_pct", 0), reverse=True)
with niah_path.open("w", encoding="utf-8") as f:
    for p in niah[:30]:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
print(f"Created {niah_path} ({len(niah[:30])} pairs)")
