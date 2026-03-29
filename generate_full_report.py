import json
from pathlib import Path

def analyze_jsonl(filepath):
    faith_scores, zeros, hallucination_rates = [], 0, []
    gen_tokens, gen_costs, latencies = [], [], []
    precisions, recalls, bertscores = [], [], []
    
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            
            f_score = r.get('faithfulness')
            if f_score is not None:
                faith_scores.append(f_score)
                if f_score == 0.0: zeros += 1
                
            gen_tokens.append(r.get('gen_input_tokens', r.get('input_tokens', 0)) or 0)
            gen_costs.append(r.get('gen_cost_usd', r.get('total_cost_usd', r.get('cost_usd', 0.0))) or 0.0)
            latencies.append(r.get('latency_s', 0.0) or 0.0)
            
            p = r.get('context_precision')
            if p is not None: precisions.append(p)
                
            rc = r.get('context_recall')
            if rc is not None: recalls.append(rc)
                
            b = r.get('bertscore_f1')
            if b is not None: bertscores.append(b)
                
            h = r.get('hallucination_rate')
            if h is not None: hallucination_rates.append(h)

    def avg(lst): return sum(lst) / len(lst) if lst else 0
    return {
        "count": len(faith_scores),
        "faith_avg": avg(faith_scores),
        "zeros": zeros,
        "halluc_avg": avg(hallucination_rates),
        "tok_avg": int(avg(gen_tokens)),
        "cost_avg": avg(gen_costs),
        "lat_avg": avg(latencies),
        "prec_avg": avg(precisions),
        "rec_avg": avg(recalls),
        "bert_avg": avg(bertscores),
        "cost_total": sum(gen_costs),
        "lat_total": sum(latencies),
        "tok_total": sum(gen_tokens)
    }

def get_best_file(pdir: Path, pattern="run_*.jsonl"):
    # Try *_bs.jsonl first
    bs_files = list(pdir.glob(pattern.replace(".jsonl", "_bs.jsonl")))
    if bs_files: return bs_files[0]
    
    # Otherwise try standard pattern
    std_files = list(pdir.glob(pattern))
    if std_files: return std_files[0]
    return None

base = Path(r"c:\Users\miqba\projects\LC vs RAG benchmark\results")

out_lines = []
out_lines.append("--- PHASE 1 ---")
for cond in ["simple_rag", "advanced_rag", "lc"]:
    p = get_best_file(base / "phase1" / cond)
    if p: out_lines.append(f"{cond.upper()}: {analyze_jsonl(p)}")

out_lines.append("\n--- PHASE 2 ---")
for model in ["gemini_flash", "gpt4o"]:
    for cond in ["simple_rag", "advanced_rag", "lc"]:
        pdir = base / "phase2" / model / cond
        p = get_best_file(pdir, "results_clean.jsonl")
        if p: out_lines.append(f"{model}/{cond}: {analyze_jsonl(p)}")

out_lines.append("\n--- ABLATION ---")
conditions = [
    ("1_Baseline", "simple_rag_baseline"),
    ("2_QueryRewrite", "plus_query_rewrite"),
    ("3_MetadataFilter", "plus_metadata_filter"),
    ("4_HybridSearch", "plus_hybrid_search"),
    ("5_Reranking", "plus_reranking"),
    ("6_Full", "full_advanced_rag")
]
for name, folder in conditions:
    pdir = base / "ablation" / folder
    p = get_best_file(pdir, "results_clean.jsonl")
    if p: out_lines.append(f"{name}: {analyze_jsonl(p)}")

out_lines.append("\n--- ADDITIONAL ---")
for sub in ["niah", "chunking_comparison/simple_rag_fixed", "chunking_comparison/simple_rag_section", "knowledge_update"]:
    pdir = base / "additional" / sub
    if pdir.exists():
        for cond in pdir.iterdir():
            if cond.is_dir():
                p = get_best_file(cond)
                if p: out_lines.append(f"{sub.upper()}_{cond.name.upper()}: {analyze_jsonl(p)}")

out_path = Path(r"c:\Users\miqba\projects\LC vs RAG benchmark\report_totals.txt")
out_path.write_text("\n".join(out_lines), encoding="utf-8")
print(f"Final Metrics with BERTScore written to {out_path}")
