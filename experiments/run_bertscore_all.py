from pathlib import Path
from experiments.pipeline.batch_evaluator import add_bertscore

results_dir = Path("results")
all_jsonl = list(results_dir.rglob("*.jsonl"))

# Filter to only the files that contain actual data we care about
# We'll skip raw/error files and focus on run_*.jsonl or results_clean.jsonl
targets = []
for p in all_jsonl:
    if "_raw" in str(p) or "errors" in str(p):
        continue
    if p.name.startswith("run_") or p.name == "results_clean.jsonl":
        targets.append(p)

print(f"Found {len(targets)} result files to process for BERTScore.")

for jsonl in targets:
    print(f"Processing: {jsonl}...")
    try:
        add_bertscore(jsonl)
        print(f"  -> Done.")
    except Exception as e:
        print(f"  -> Error on {jsonl}: {e}")

print("BERTScore evaluation complete for all result files.")
