"""
Automated LLM Reviewer for QA dataset validation.

Reviews a sample of draft QA pairs using "Elite" models (Claude, GPT, Gemini).
Output is used for Fleiss' Kappa IAA calculation.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict

from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import openai
import anthropic

from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)

class ReviewResponse(BaseModel):
    status: str = Field(description="One of: accepted, modified, rejected")
    revised_question: str | None = Field(default=None, description="Revised question if modified")
    revised_answer: str | None = Field(default=None, description="Revised gold answer if modified")
    revised_paragraphs: List[str] | None = Field(default=None, description="Revised gold paragraphs if modified")
    reasoning: str = Field(description="Explanation for the decision")

SYSTEM_PROMPT = """
You are a legal expert reviewing a QA dataset based on Indonesian Constitutional Court verdicts.
Your task is to review a Question-Answer pair and decide if it is accurate, helpful, and grounded in the provided verdict text.

Actions:
- ACCEPT: The question and answer are perfect.
- MODIFY: The question or answer needs minor refinement (provide the revised version).
- REJECT: The question is invalid, nonsensical, or not grounded in the verdict.

Rules for Legal Accuracy:
1. Questions must be specific to the verdict.
2. Answers must be legally sound and grounded in the 'Pertimbangan Hukum' (Legal Considerations) or 'Amar' (Judgment).
3. Supporting paragraphs must be relevant.
"""

def review_with_gemini(client: genai.Client, model_id: str, prompt: str) -> ReviewResponse:
    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=ReviewResponse,
            temperature=0.0,
        )
    )
    return ReviewResponse.model_validate_json(response.text)

def review_with_openai(client: openai.OpenAI, model_id: str, prompt: str) -> ReviewResponse:
    response = client.beta.chat.completions.parse(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        response_format=ReviewResponse,
        temperature=0.0,
    )
    return response.choices[0].message.parsed

def review_with_anthropic(client: anthropic.Anthropic, model_id: str, prompt: str) -> ReviewResponse:
    # Anthropic doesn't have native Pydantic parsing yet, use manual extraction
    response = client.messages.create(
        model=model_id,
        max_tokens=1024,
        system=SYSTEM_PROMPT + "\nOutput valid JSON matching this schema: " + json.dumps(ReviewResponse.model_json_schema()),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    # Basic JSON cleaning
    text = response.content[0].text
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    return ReviewResponse.model_validate_json(text)

def run_automated_review():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model ID to use (gemini/gpt/claude)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--sample_size", type=int, default=1000, help="Number of pairs to review (default 1000 for 100% coverage)")
    parser.add_argument("--resume", action="store_true", help="Resume from last reviewed ID")
    args = parser.parse_args()

    drafts_path = cfg.paths.qa_dir / "qa_drafts_raw.jsonl"
    output_path = Path(args.output)
    
    # Load drafts
    with open(drafts_path, encoding="utf-8") as f:
        drafts = [json.loads(line) for line in f if line.strip()]

    # Skip already reviewed if resuming
    reviewed_ids = set()
    if args.resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                reviewed_ids.add(json.loads(line)["question_id"])
    
    pending = [d for d in drafts if d["question_id"] not in reviewed_ids][:args.sample_size]
    
    if not pending:
        log.info("No pending drafts to review.")
        return

    # Clients
    if "gemini" in args.model.lower():
        client = genai.Client(api_key=cfg.keys["google_api_key"])
        reviewer_fn = lambda p: review_with_gemini(client, args.model, p)
    elif "gpt" in args.model.lower():
        client = openai.OpenAI(api_key=cfg.keys["openai_api_key"])
        reviewer_fn = lambda p: review_with_openai(client, args.model, p)
    elif "claude" in args.model.lower():
        client = anthropic.Anthropic(api_key=cfg.keys["anthropic_api_key"])
        reviewer_fn = lambda p: review_with_anthropic(client, args.model, p)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    log.info(f"Starting review with {args.model} | Sample: {len(pending)}")
    
    with open(output_path, "a", encoding="utf-8") as f:
        for i, pair in enumerate(pending):
            prompt = (
                f"Verdict Text (Supporting Context):\n{pair.get('gold_paragraphs', ['N/A'])}\n\n"
                f"Draft Question: {pair['question']}\n"
                f"Draft Answer: {pair['gold_answer']}\n"
            )
            
            try:
                review = reviewer_fn(prompt)
                
                # Merge review into pair
                pair["status"] = review.status
                pair["reviewer_model"] = args.model
                pair["reasoning"] = review.reasoning
                if review.status == "modified":
                    if review.revised_question: pair["question"] = review.revised_question
                    if review.revised_answer: pair["gold_answer"] = review.revised_answer
                    if review.revised_paragraphs: pair["gold_paragraphs"] = review.revised_paragraphs
                
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                f.flush()
                log.info(f"[{i+1}/{len(pending)}] Reviewed {pair['question_id']}: {review.status}")
                
            except Exception as e:
                log.error(f"Failed to review {pair['question_id']}: {e}")

if __name__ == "__main__":
    run_automated_review()
