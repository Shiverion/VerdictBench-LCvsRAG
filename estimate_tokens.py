import os
import json
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv()

pairs = [json.loads(l) for l in Path('data/qa_dataset/qa_pairs_full.jsonl').read_text(encoding='utf-8').splitlines()]
client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
total_input = 0
counts = []

for idx, p in enumerate(pairs[:50]): # testing first 50
    # The real evaluator uses the FULL cleaned verdict text for 'lc'
    v_path = Path('data/processed/cleaned') / f"{p['verdict_id']}.txt"
    if not v_path.exists():
        continue
    context = v_path.read_text(encoding='utf-8')
    prompt = f"""Question: {p['question']}
Answer: {p['gold_answer']}
Source: {context[:400000]} # Limit to 100k tokens roughly

Rate faithfulness 0-1."""
    
    result = client.models.count_tokens(model='models/gemini-2.5-flash', contents=prompt)
    total_input += result.total_tokens
    counts.append(result.total_tokens)
    print(f"{p['question_id']}: {result.total_tokens} tokens")

avg = total_input / len(counts)
max_tokens = max(counts)
p95_tokens = sorted(counts)[int(len(counts)*0.95)]
projected_1050 = avg * 1050
print(f"\nAvg per call: {avg:.0f} tokens")
print(f"Max: {max_tokens} | P95: {p95_tokens}")
print(f"Projected total input (1050 calls): {projected_1050:,.0f} tokens")
print(f"Input cost @ $2/1M: ${projected_1050/1e6*2:.2f} = Rp{projected_1050/1e6*2*16933:,.0f}")
