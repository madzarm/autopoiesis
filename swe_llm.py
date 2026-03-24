"""LLM utility for SWE-bench — supports both OpenAI and Anthropic with cost tracking."""

import os
import time
import threading
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential

# ── Model definitions ────────────────────────────────────────────────────────

# OpenAI models
GPT_NANO = "gpt-5.4-nano-2026-03-17"   # Cheapest, fast
GPT_MINI = "gpt-5.4-mini-2026-03-17"   # Good balance
GPT_STRONG = "gpt-5.4-2026-03-05"      # Strongest OpenAI

# Claude models
SONNET = "claude-sonnet-4-20250514"
HAIKU = "claude-haiku-4-5-20251001"
OPUS = "claude-opus-4-0-20250514"

# Default agent model — gpt-5.4-nano is cheapest
AGENT_MODEL = GPT_NANO

# Cost per 1M tokens
COST_PER_1M = {
    GPT_NANO: {"input": 0.10, "output": 0.40},
    GPT_MINI: {"input": 0.40, "output": 1.60},
    GPT_STRONG: {"input": 2.0, "output": 8.0},
    SONNET: {"input": 3.0, "output": 15.0},
    HAIKU: {"input": 0.80, "output": 4.0},
    OPUS: {"input": 15.0, "output": 75.0},
}

# ── Clients ──────────────────────────────────────────────────────────────────

_anthropic_client = None
_openai_client = None
_total_cost: float = 0.0
_call_count: int = 0
_total_input_tokens: int = 0
_total_output_tokens: int = 0
_cost_lock = threading.Lock()


def _is_openai_model(model: str) -> bool:
    return model.startswith("gpt-") or model.startswith("o")


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "")
        )
    return _anthropic_client


def _get_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    return _openai_client


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    costs = COST_PER_1M.get(model, {"input": 1.0, "output": 4.0})
    return (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1_000_000


# ── Main API ─────────────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(min=2, max=120),
    retry=lambda retry_state: _should_retry(retry_state),
)
def call_llm(
    prompt: str,
    system: str = "",
    model: str = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    messages: list = None,
) -> dict:
    """Call LLM (auto-routes to OpenAI or Anthropic based on model name)."""
    if model is None:
        model = AGENT_MODEL

    if _is_openai_model(model):
        return _call_openai(prompt, system, model, temperature, max_tokens, messages)
    else:
        return _call_anthropic(prompt, system, model, temperature, max_tokens, messages)


def call_llm_multi_turn(
    messages: list,
    system: str = "",
    model: str = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
) -> dict:
    """Multi-turn conversation."""
    if model is None:
        model = AGENT_MODEL
    return call_llm("", system=system, model=model, temperature=temperature,
                     max_tokens=max_tokens, messages=messages)


# ── OpenAI backend ───────────────────────────────────────────────────────────

def _call_openai(prompt, system, model, temperature, max_tokens, messages):
    global _total_cost, _call_count, _total_input_tokens, _total_output_tokens

    client = _get_openai()

    if messages is None:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
    elif system:
        # Prepend system message if not already there
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": system}] + messages

    # Model-specific API differences
    uses_new_api = model.startswith("o") or "gpt-5" in model
    kwargs = {"model": model, "messages": messages, "timeout": 120}

    if uses_new_api:
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens

    if model.startswith("o"):
        # o-series: developer role instead of system
        for m in kwargs["messages"]:
            if m["role"] == "system":
                m["role"] = "developer"
    else:
        kwargs["temperature"] = temperature

    start = time.time()
    response = client.chat.completions.create(**kwargs)
    elapsed = time.time() - start

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Empty response from API")

    usage = response.usage
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
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


# ── Anthropic backend ────────────────────────────────────────────────────────

def _call_anthropic(prompt, system, model, temperature, max_tokens, messages):
    global _total_cost, _call_count, _total_input_tokens, _total_output_tokens

    client = _get_anthropic()

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


# ── Retry logic ──────────────────────────────────────────────────────────────

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


# ── Stats ────────────────────────────────────────────────────────────────────

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
