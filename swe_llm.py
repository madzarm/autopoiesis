"""LLM utility for SWE-bench — Anthropic Claude client with cost tracking."""

import os
import time
import threading
from typing import Optional
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

# Claude models
SONNET = "claude-sonnet-4-20250514"       # Meta-model + strong agent backbone
HAIKU = "claude-haiku-4-5-20251001"       # Cheap agent backbone ($0.80/$4 per 1M)
OPUS = "claude-opus-4-0-20250514"         # Strongest reasoning (expensive)

# Default agent model
AGENT_MODEL = SONNET  # Claude Sonnet 4 — comparable to EvoMAS's Claude-3.5-Sonnet

# Cost per 1M tokens
COST_PER_1M = {
    SONNET: {"input": 3.0, "output": 15.0},
    HAIKU: {"input": 0.80, "output": 4.0},
    OPUS: {"input": 15.0, "output": 75.0},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
}

_client: Optional[anthropic.Anthropic] = None
_total_cost: float = 0.0
_call_count: int = 0
_total_input_tokens: int = 0
_total_output_tokens: int = 0
_cost_lock = threading.Lock()


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    costs = COST_PER_1M.get(model, {"input": 3.0, "output": 15.0})
    return (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1_000_000


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(min=2, max=120),
    retry=lambda retry_state: _should_retry(retry_state),
)
def call_llm(
    prompt: str,
    system: str = "",
    model: str = SONNET,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    messages: list = None,
) -> dict:
    """Call Claude and return response with metadata."""
    global _total_cost, _call_count, _total_input_tokens, _total_output_tokens

    client = get_client()

    if messages is None:
        messages = [{"role": "user", "content": prompt}]

    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if system:
        kwargs["system"] = system

    start = time.time()
    response = client.messages.create(**kwargs)
    elapsed = time.time() - start

    content = response.content[0].text if response.content else ""
    if not content:
        raise ValueError("Empty response from API")

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost = estimate_cost(model, input_tokens, output_tokens)

    with _cost_lock:
        _total_cost += cost
        _call_count += 1
        _total_input_tokens += input_tokens
        _total_output_tokens += output_tokens

    return {
        "content": content,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost,
        "elapsed_s": elapsed,
    }


def call_llm_multi_turn(
    messages: list,
    system: str = "",
    model: str = SONNET,
    temperature: float = 0.0,
    max_tokens: int = 8192,
) -> dict:
    """Multi-turn conversation with Claude."""
    return call_llm("", system=system, model=model, temperature=temperature,
                     max_tokens=max_tokens, messages=messages)


def _should_retry(retry_state):
    exc = retry_state.outcome.exception()
    if exc is None:
        return False
    exc_str = str(type(exc).__name__) + ": " + str(exc)
    if any(s in exc_str.lower() for s in ["rate_limit", "429", "timeout", "connection",
                                           "server_error", "500", "502", "503", "529",
                                           "overloaded", "empty response"]):
        return True
    if "APIError" in type(exc).__name__ or "APIConnectionError" in type(exc).__name__:
        return True
    if any(s in exc_str.lower() for s in ["authentication", "401", "403"]):
        return False
    return True


def get_session_cost() -> float:
    return _total_cost

def get_call_count() -> int:
    return _call_count

def get_session_stats() -> dict:
    return {
        "total_cost_usd": round(_total_cost, 4),
        "call_count": _call_count,
        "total_input_tokens": _total_input_tokens,
        "total_output_tokens": _total_output_tokens,
    }

def reset_cost_tracking():
    global _total_cost, _call_count, _total_input_tokens, _total_output_tokens
    _total_cost = 0.0
    _call_count = 0
    _total_input_tokens = 0
    _total_output_tokens = 0
