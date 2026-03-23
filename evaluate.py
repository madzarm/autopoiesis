"""Evaluation harness — runs benchmarks and returns scores.

Supports parallel evaluation via ThreadPoolExecutor for ~10x speedup.
"""

import json
import re
import random
from typing import Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from llm import call_llm, CHEAP, get_session_cost, reset_cost_tracking

# Max concurrent LLM calls for parallel eval
MAX_WORKERS = 16


def extract_number(text: str) -> Optional[float]:
    """Extract the final number from a response (for math benchmarks)."""
    # Look for #### pattern first (GSM8K style)
    match = re.search(r'####\s*([\-\d,\.]+)', text)
    if match:
        return float(match.group(1).replace(',', ''))
    # Look for \boxed{} pattern (MATH style)
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        try:
            return float(match.group(1).replace(',', ''))
        except ValueError:
            return None
    # Fall back to last number in text
    numbers = re.findall(r'[\-]?\d+\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            return None
    return None


def load_gsm8k(split: str = "test", n: Optional[int] = None, seed: int = 42) -> list[dict]:
    """Load GSM8K benchmark samples."""
    ds = load_dataset("openai/gsm8k", "main", split=split)
    samples = []
    for item in ds:
        answer_text = item["answer"]
        match = re.search(r'####\s*([\-\d,\.]+)', answer_text)
        if match:
            gold = float(match.group(1).replace(',', ''))
        else:
            continue
        samples.append({
            "question": item["question"],
            "gold_answer": gold,
            "gold_reasoning": answer_text,
        })
    if n is not None and n < len(samples):
        rng = random.Random(seed)
        samples = rng.sample(samples, n)
    return samples


def load_drop(split: str = "validation", n: Optional[int] = None, seed: int = 42) -> list[dict]:
    """Load DROP benchmark samples."""
    ds = load_dataset("ucinlp/drop", split=split)
    samples = []
    for item in ds:
        answers = item["answers_spans"]
        if not answers or not answers["spans"]:
            continue
        samples.append({
            "passage": item["passage"],
            "question": item["question"],
            "gold_answers": answers["spans"],
        })
    if n is not None and n < len(samples):
        rng = random.Random(seed)
        samples = rng.sample(samples, n)
    return samples


def load_mgsm(lang: str = "en", n: Optional[int] = None, seed: int = 42) -> list[dict]:
    """Load MGSM benchmark samples."""
    ds = load_dataset("juletxara/mgsm", lang, split="test")
    samples = []
    for item in ds:
        samples.append({
            "question": item["question"],
            "gold_answer": float(item["answer_number"]),
        })
    if n is not None and n < len(samples):
        rng = random.Random(seed)
        samples = rng.sample(samples, n)
    return samples


def _eval_single_math(agent_fn, sample, idx):
    """Evaluate a single math sample. Thread-safe."""
    try:
        response = agent_fn(sample["question"])
        predicted = extract_number(response)
        gold = sample["gold_answer"]
        is_correct = predicted is not None and abs(predicted - gold) < 1e-6
        return {
            "idx": idx,
            "correct": is_correct,
            "predicted": predicted,
            "gold": gold,
            "question": sample["question"][:200],
        }
    except Exception as e:
        return {
            "idx": idx,
            "correct": False,
            "error": str(e),
            "question": sample.get("question", "")[:200],
        }


def evaluate_math_accuracy(
    agent_fn: Callable[[str], str],
    samples: list[dict],
    benchmark_name: str = "gsm8k",
    parallel: bool = True,
    max_workers: int = MAX_WORKERS,
) -> dict:
    """Evaluate agent on math benchmark (GSM8K, MGSM).

    Uses parallel execution by default for ~10x speedup.
    """
    reset_cost_tracking()
    total = len(samples)
    details = []

    if parallel and total > 1:
        with ThreadPoolExecutor(max_workers=min(max_workers, total)) as executor:
            futures = {
                executor.submit(_eval_single_math, agent_fn, sample, i): i
                for i, sample in enumerate(samples)
            }
            for future in as_completed(futures):
                details.append(future.result())
    else:
        for i, sample in enumerate(samples):
            details.append(_eval_single_math(agent_fn, sample, i))

    # Sort by index for consistent ordering
    details.sort(key=lambda x: x["idx"])
    correct = sum(1 for d in details if d.get("correct", False))

    score = correct / total if total > 0 else 0.0
    return {
        "benchmark": benchmark_name,
        "score": round(score * 100, 2),
        "correct": correct,
        "total": total,
        "cost_usd": round(get_session_cost(), 4),
        "details": details,
    }


def normalize_answer(s: str) -> str:
    """Normalize answer string for comparison."""
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def compute_f1(prediction: str, gold: str) -> float:
    """Compute token-level F1 between prediction and gold."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not gold_tokens:
        return float(not pred_tokens)
    if not pred_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _eval_single_drop(agent_fn, sample, idx):
    """Evaluate a single DROP sample. Thread-safe."""
    try:
        response = agent_fn(sample["passage"], sample["question"])
        max_f1 = max(
            compute_f1(response, gold)
            for gold in sample["gold_answers"]
        )
        return {
            "idx": idx,
            "f1": max_f1,
            "predicted": response[:200],
        }
    except Exception as e:
        return {"idx": idx, "f1": 0.0, "error": str(e)}


def evaluate_drop_f1(
    agent_fn: Callable[[str, str], str],
    samples: list[dict],
    parallel: bool = True,
    max_workers: int = MAX_WORKERS,
) -> dict:
    """Evaluate agent on DROP (F1 metric). Parallel by default."""
    reset_cost_tracking()
    details = []

    if parallel and len(samples) > 1:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(samples))) as executor:
            futures = {
                executor.submit(_eval_single_drop, agent_fn, sample, i): i
                for i, sample in enumerate(samples)
            }
            for future in as_completed(futures):
                details.append(future.result())
    else:
        for i, sample in enumerate(samples):
            details.append(_eval_single_drop(agent_fn, sample, i))

    details.sort(key=lambda x: x["idx"])
    f1_scores = [d["f1"] for d in details]

    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    return {
        "benchmark": "drop",
        "score": round(avg_f1 * 100, 2),
        "total": len(samples),
        "cost_usd": round(get_session_cost(), 4),
        "details": details,
    }
