"""LLM utility module — OpenAI API client with cost tracking."""

import os
import time
import threading
from typing import Optional
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Models by tier (per program.md)
STRONG = "gpt-5.4-2026-03-05"  # Meta-agent / search algorithm
MID = "o4-mini"  # Inner agents being designed
CHEAP = "gpt-5.4-nano-2026-03-17"  # Evaluation / high-volume calls

# Approximate cost per 1M tokens (input/output) — for tracking
COST_PER_1M = {
    "gpt-5.4-2026-03-05": {"input": 2.0, "output": 8.0},
    "gpt-5.4-nano-2026-03-17": {"input": 0.10, "output": 0.40},
    "gpt-5.4-mini-2026-03-17": {"input": 0.40, "output": 1.60},
    "o4-mini": {"input": 1.10, "output": 4.40},
    "gpt-4.1": {"input": 2.0, "output": 8.0},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

_client: Optional[OpenAI] = None
_total_cost: float = 0.0
_call_count: int = 0
_cost_lock = threading.Lock()


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        _client = OpenAI(api_key=api_key)
    return _client


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    costs = COST_PER_1M.get(model, {"input": 1.0, "output": 4.0})
    return (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1_000_000


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(min=2, max=60),
    retry=lambda retry_state: _should_retry(retry_state),
)
def call_llm(
    prompt: str,
    system: str = "",
    model: str = CHEAP,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    json_mode: bool = False,
) -> dict:
    """Call the LLM and return response with metadata.

    Retries up to 5 times with exponential backoff on:
    - API errors (network, 500, 503)
    - Rate limits (429) — with longer backoff
    - Empty responses
    - Timeouts

    Returns:
        dict with keys: content, model, input_tokens, output_tokens, cost_usd
    """
    global _total_cost, _call_count

    client = get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # Models that require max_completion_tokens instead of max_tokens
    uses_new_api = model.startswith("o") or "gpt-5" in model

    kwargs = {
        "model": model,
        "messages": messages,
        "timeout": 120,  # 2-minute timeout per call
    }

    if uses_new_api:
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens

    if model.startswith("o"):
        # o-series uses developer messages instead of system
        if system and messages[0]["role"] == "system":
            messages[0]["role"] = "developer"
    else:
        kwargs["temperature"] = temperature

    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    start = time.time()
    response = client.chat.completions.create(**kwargs)
    elapsed = time.time() - start

    usage = response.usage
    cost = estimate_cost(model, usage.prompt_tokens, usage.completion_tokens)

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Empty response from API")

    with _cost_lock:
        _total_cost += cost
        _call_count += 1

    return {
        "content": content,
        "model": model,
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "cost_usd": cost,
        "elapsed_s": elapsed,
    }


def _should_retry(retry_state):
    """Determine if we should retry based on the exception type."""
    exc = retry_state.outcome.exception()
    if exc is None:
        return False  # Success, don't retry
    exc_str = str(type(exc).__name__) + ": " + str(exc)
    # Always retry on these
    if any(s in exc_str.lower() for s in ["rate_limit", "429", "timeout", "connection",
                                           "server_error", "500", "502", "503", "529",
                                           "empty response", "overloaded"]):
        return True
    # Retry on generic API errors
    if "APIError" in type(exc).__name__ or "APIConnectionError" in type(exc).__name__:
        return True
    # Don't retry on auth errors, invalid requests, etc.
    if any(s in exc_str.lower() for s in ["authentication", "401", "403", "invalid"]):
        return False
    # Default: retry on unknown errors
    return True


def get_session_cost() -> float:
    return _total_cost


def get_call_count() -> int:
    return _call_count


def reset_cost_tracking():
    global _total_cost, _call_count
    _total_cost = 0.0
    _call_count = 0
