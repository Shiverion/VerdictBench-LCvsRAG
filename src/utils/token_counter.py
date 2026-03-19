"""
Token estimation for cost tracking.
- OpenAI models: tiktoken (exact)
- Gemini models: character-based approximation (4 chars ≈ 1 token)
"""

from __future__ import annotations

from functools import lru_cache

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


_GEMINI_CHARS_PER_TOKEN = 4.0


@lru_cache(maxsize=8)
def _get_encoding(model: str):
    """Cache tiktoken encodings — expensive to initialize."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str) -> int:
    """
    Estimate input token count for a given model.

    Args:
        text:  The text to count tokens for.
        model: Model name (e.g. 'gpt-4o', 'gemini-2.5-flash-preview-04-17').

    Returns:
        Estimated token count.
    """
    if not text:
        return 0

    if model.startswith("gpt") or model.startswith("o1"):
        if _TIKTOKEN_AVAILABLE:
            enc = _get_encoding(model)
            return len(enc.encode(text))
        # Fallback if tiktoken not installed
        return int(len(text) / 4.0)

    # Gemini and other models: character approximation
    return int(len(text) / _GEMINI_CHARS_PER_TOKEN)


def estimate_cost(input_tokens: int, model: str, pricing: dict[str, float]) -> float:
    """
    Estimate input token cost in USD.

    Args:
        input_tokens: Number of input tokens.
        model:        Model name.
        pricing:      Dict of {model_name: price_per_1M_tokens}.

    Returns:
        Cost in USD.
    """
    price_per_1m = pricing.get(model, 0.0)
    return (input_tokens / 1_000_000) * price_per_1m
