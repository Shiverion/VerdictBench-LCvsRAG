from pathlib import Path
from experiments.pipeline.batch_evaluator import add_bertscore

target = Path("results/phase2/gpt4o/simple_rag/results_clean.jsonl")
print(f"Processing: {target}...")
try:
    add_bertscore(target)
    print(f"  -> Done.")
except Exception as e:
    print(f"  -> Error on {target}: {e}")

print("Recovery evaluation complete.")
