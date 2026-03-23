"""LLM utility module — OpenAI API client with cost tracking."""

import os
import time
import threading
from typing import Optional
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Models by tier
STRONG = "gpt-4.1"  # Meta-agent / search algorithm
MID = "o4-mini"  # Inner agents being designed
CHEAP = "gpt-4.1-nano"  # Evaluation / high-volume calls

# Approximate cost per 1M tokens (input/output) — for tracking
COST_PER_1M = {
    "gpt-4.1": {"input": 2.0, "output": 8.0},
    "o4-mini": {"input": 1.10, "output": 4.40},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
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


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
def call_llm(
    prompt: str,
    system: str = "",
    model: str = CHEAP,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    json_mode: bool = False,
) -> dict:
    """Call the LLM and return response with metadata.

    Returns:
        dict with keys: content, model, input_tokens, output_tokens, cost_usd
    """
    global _total_cost, _call_count

    client = get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if model.startswith("o"):
        # o-series models don't support temperature or system messages
        kwargs.pop("max_tokens")
        kwargs["max_completion_tokens"] = max_tokens
        if not system:
            pass  # no system message needed
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

    with _cost_lock:
        _total_cost += cost
        _call_count += 1

    return {
        "content": response.choices[0].message.content,
        "model": model,
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "cost_usd": cost,
        "elapsed_s": elapsed,
    }


def get_session_cost() -> float:
    return _total_cost


def get_call_count() -> int:
    return _call_count


def reset_cost_tracking():
    global _total_cost, _call_count
    _total_cost = 0.0
    _call_count = 0
