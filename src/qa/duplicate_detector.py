"""
Semantic Duplicate Detector
===========================
Identifies semantically identical or highly overlapping QA pairs 
within the same verdict using `text-embedding-004`. 

This acts as a supplementary automated check to warn human annotators
of redundant questions before they are approved into the final dataset.
"""

import json
from pathlib import Path
import numpy as np

from google import genai
from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))

def run_duplicate_check(dataset_path: Path, threshold: float = 0.88):
    if not dataset_path.exists():
        log.error(f"Dataset not found: {dataset_path}")
        return

    pairs = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    
    log.info(f"Loaded {len(pairs)} pairs for duplicate detection.")
    
    texts = [p["question"] for p in pairs]
    ids = [p["question_id"] for p in pairs]
    
    api_key = cfg.keys.get("google_api_key")
    if not api_key:
        # Fallback to os env if dict notation missed
        import os
        api_key = os.getenv("GOOGLE_API_KEY")

    client = genai.Client(api_key=api_key)
    
    import time
    batch_size = 5
    all_embeddings = []
    
    log.info("Generating embeddings for all drafted questions (with rate limiting)...")
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        # Exponential backoff for rate limits
        retries = 3
        while retries > 0:
            try:
                response = client.models.embed_content(
                    model=cfg.models.embedding_model,
                    contents=batch
                )
                if hasattr(response, 'embeddings'):
                    all_embeddings.extend([e.values for e in response.embeddings])
                time.sleep(2.0)
                break
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    log.warning(f"Rate limited. Waiting 10s... (Retries left: {retries})")
                    time.sleep(10)
                    retries -= 1
                else:
                    raise e

    if len(all_embeddings) != len(texts):
        log.error(f"Failed to embed all queries. Embedded {len(all_embeddings)}/{len(texts)}")
        return
        
    log.info("Computing pairwise cosine similarity within verdicts...")
    suspected_duplicates = set()
    pairs_flagged = 0
    
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            # Only flag duplicates WITHIN the same verdict case
            if pairs[i]["verdict_id"] == pairs[j]["verdict_id"]:
                sim = cosine_similarity(all_embeddings[i], all_embeddings[j])
                if sim >= threshold:
                    suspected_duplicates.add(ids[i])
                    suspected_duplicates.add(ids[j])
                    pairs_flagged += 1
                    log.warning(f"[Overlap {sim:.2f}] {ids[i]} <-> {ids[j]} in {pairs[i]['verdict_id']}")
                    log.warning(f"  Q1: {texts[i]}")
                    log.warning(f"  Q2: {texts[j]}")

    log.info(f"--- Semantic Duplicate Report ---")
    log.info(f"Pairs compared: {len(texts)}")
    log.info(f"Duplicate pairs found: {pairs_flagged}")
    log.info(f"Unique flagged IDs: {len(suspected_duplicates)}")
    
    flags_path = dataset_path.parent / "duplicate_flags.json"
    with open(flags_path, "w", encoding="utf-8") as f:
        json.dump(list(suspected_duplicates), f)
    log.info(f"Saved {len(suspected_duplicates)} duplicate flags to {flags_path}")

def main():
    dataset = cfg.paths.qa_dir / "qa_drafts_raw.jsonl"
    run_duplicate_check(dataset, threshold=0.88)

if __name__ == "__main__":
    main()
