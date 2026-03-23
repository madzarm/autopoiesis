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
    try:
        ds = load_dataset("juletxara/mgsm", lang, split="test", trust_remote_code=True)
    except Exception:
        # Fallback: use the HuggingFace MGSM dataset with different config
        ds = load_dataset("google/mgsm", lang, split="test", trust_remote_code=True)
    samples = []
    for item in ds:
        # Handle different column names across dataset versions
        question = item.get("question", item.get("input", ""))
        answer = item.get("answer_number", item.get("target", item.get("answer", "")))
        try:
            gold = float(answer)
        except (ValueError, TypeError):
            # Try to extract number from answer string
            match = re.search(r'[\-]?\d+\.?\d*', str(answer))
            if match:
                gold = float(match.group())
            else:
                continue
        samples.append({
            "question": question,
            "gold_answer": gold,
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


def extract_text_answer(response: str) -> str:
    """Extract the text answer from agent response. Looks for #### marker or uses last line."""
    # Look for #### marker
    match = re.search(r'####\s*(.+)', response)
    if match:
        return match.group(1).strip()
    # Use last non-empty line
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    return lines[-1] if lines else response.strip()


def _eval_single_drop(agent_fn, sample, idx):
    """Evaluate a single DROP sample. Thread-safe."""
    try:
        response = agent_fn(sample["passage"], sample["question"])
        # Extract just the answer portion for F1 comparison
        predicted = extract_text_answer(response)
        max_f1 = max(
            compute_f1(predicted, gold)
            for gold in sample["gold_answers"]
        )
        return {
            "idx": idx,
            "f1": max_f1,
            "predicted": predicted[:200],
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


# ─── ARC-Challenge ────────────────────────────────────

def load_arc(split: str = "test", n: Optional[int] = None, seed: int = 42) -> list[dict]:
    """Load ARC-Challenge benchmark samples."""
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    samples = []
    for item in ds:
        choices = item["choices"]
        labels = choices["label"]
        texts = choices["text"]
        answer_key = item["answerKey"]
        # Build choice string
        choice_str = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
        samples.append({
            "question": item["question"],
            "choices": choice_str,
            "choices_labels": labels,
            "choices_texts": texts,
            "gold_answer": answer_key,
        })
    if n is not None and n < len(samples):
        rng = random.Random(seed)
        samples = rng.sample(samples, n)
    return samples


def _eval_single_arc(agent_fn, sample, idx):
    """Evaluate a single ARC sample. Thread-safe."""
    try:
        response = agent_fn(sample["question"], sample["choices"])
        # Extract the answer letter
        predicted = extract_answer_letter(response, sample["choices_labels"])
        gold = sample["gold_answer"]
        is_correct = predicted == gold
        return {
            "idx": idx,
            "correct": is_correct,
            "predicted": predicted,
            "gold": gold,
        }
    except Exception as e:
        return {"idx": idx, "correct": False, "error": str(e)}


def extract_answer_letter(text: str, valid_labels: list[str]) -> Optional[str]:
    """Extract answer letter (A, B, C, D, etc.) from response."""
    text_upper = text.upper()
    # Look for "The answer is X" pattern
    match = re.search(r'(?:answer|correct)\s+is\s+\(?([A-E])\)?', text_upper)
    if match and match.group(1) in valid_labels:
        return match.group(1)
    # Look for "#### X" pattern
    match = re.search(r'####\s*\(?([A-E])\)?', text_upper)
    if match and match.group(1) in valid_labels:
        return match.group(1)
    # Look for standalone letter at end
    match = re.search(r'\b([A-E])\b\s*$', text_upper.strip())
    if match and match.group(1) in valid_labels:
        return match.group(1)
    # First mentioned valid label
    for label in valid_labels:
        if re.search(rf'\b{label}\b', text_upper):
            return label
    return None


def evaluate_arc_accuracy(
    agent_fn: Callable[[str, str], str],
    samples: list[dict],
    parallel: bool = True,
    max_workers: int = MAX_WORKERS,
) -> dict:
    """Evaluate agent on ARC-Challenge (accuracy). agent_fn(question, choices_str) -> str."""
    reset_cost_tracking()
    details = []

    if parallel and len(samples) > 1:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(samples))) as executor:
            futures = {
                executor.submit(_eval_single_arc, agent_fn, sample, i): i
                for i, sample in enumerate(samples)
            }
            for future in as_completed(futures):
                details.append(future.result())
    else:
        for i, sample in enumerate(samples):
            details.append(_eval_single_arc(agent_fn, sample, i))

    details.sort(key=lambda x: x["idx"])
    correct = sum(1 for d in details if d.get("correct", False))
    total = len(samples)

    return {
        "benchmark": "arc_challenge",
        "score": round(correct / total * 100, 2) if total > 0 else 0.0,
        "correct": correct,
        "total": total,
        "cost_usd": round(get_session_cost(), 4),
        "details": details,
    }


# ─── MMLU ────────────────────────────────────

def load_mmlu(subject: str = "all", split: str = "test", n: Optional[int] = None, seed: int = 42) -> list[dict]:
    """Load MMLU benchmark samples."""
    ds = load_dataset("cais/mmlu", subject, split=split)
    samples = []
    labels = ["A", "B", "C", "D"]
    for item in ds:
        choices_text = item["choices"]
        choice_str = "\n".join(f"{l}. {t}" for l, t in zip(labels, choices_text))
        gold_idx = item["answer"]  # 0-3
        samples.append({
            "question": item["question"],
            "choices": choice_str,
            "choices_labels": labels,
            "choices_texts": choices_text,
            "gold_answer": labels[gold_idx],
            "subject": item.get("subject", subject),
        })
    if n is not None and n < len(samples):
        rng = random.Random(seed)
        samples = rng.sample(samples, n)
    return samples


def evaluate_mmlu_accuracy(
    agent_fn: Callable[[str, str], str],
    samples: list[dict],
    parallel: bool = True,
    max_workers: int = MAX_WORKERS,
) -> dict:
    """Evaluate agent on MMLU (accuracy). agent_fn(question, choices_str) -> str."""
    reset_cost_tracking()
    details = []

    if parallel and len(samples) > 1:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(samples))) as executor:
            futures = {
                executor.submit(_eval_single_arc, agent_fn, sample, i): i
                for i, sample in enumerate(samples)
            }
            for future in as_completed(futures):
                details.append(future.result())
    else:
        for i, sample in enumerate(samples):
            details.append(_eval_single_arc(agent_fn, sample, i))

    details.sort(key=lambda x: x["idx"])
    correct = sum(1 for d in details if d.get("correct", False))
    total = len(samples)

    return {
        "benchmark": "mmlu",
        "score": round(correct / total * 100, 2) if total > 0 else 0.0,
        "correct": correct,
        "total": total,
        "cost_usd": round(get_session_cost(), 4),
        "details": details,
    }


# ─── MATH ────────────────────────────────────

def load_math(split: str = "test", n: Optional[int] = None, seed: int = 42) -> list[dict]:
    """Load MATH benchmark."""
    ds = load_dataset("lighteval/MATH-Hard", split=split)
    samples = []
    for item in ds:
        samples.append({
            "problem": item["problem"],
            "solution": item["solution"],
            "level": item.get("level", ""),
            "type": item.get("type", ""),
        })
    if n is not None and n < len(samples):
        rng = random.Random(seed)
        samples = rng.sample(samples, n)
    return samples


def extract_math_answer(text: str) -> Optional[str]:
    """Extract answer from MATH response — handles boxed and #### formats."""
    # Look for \boxed{...}
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()
    # Look for #### format
    match = re.search(r'####\s*(.+?)(?:\n|$)', text)
    if match:
        return match.group(1).strip()
    return None


def normalize_math_answer(s: str) -> str:
    """Normalize a math answer for comparison."""
    s = s.strip()
    # Remove $ signs
    s = s.replace('$', '')
    # Remove \text{...}
    s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)
    # Remove spaces
    s = s.replace(' ', '')
    # Try to evaluate as number
    try:
        return str(float(s))
    except ValueError:
        return s.lower()


def _eval_single_math_bench(agent_fn, sample, idx):
    """Evaluate a single MATH sample."""
    try:
        response = agent_fn(sample["problem"])
        predicted = extract_math_answer(response)
        gold = extract_math_answer(sample["solution"])

        if predicted is None or gold is None:
            return {"idx": idx, "correct": False, "predicted": predicted, "gold": gold}

        is_correct = normalize_math_answer(predicted) == normalize_math_answer(gold)
        return {
            "idx": idx,
            "correct": is_correct,
            "predicted": predicted,
            "gold": gold,
        }
    except Exception as e:
        return {"idx": idx, "correct": False, "error": str(e)}


def evaluate_math_bench(
    agent_fn: Callable[[str], str],
    samples: list[dict],
    parallel: bool = True,
    max_workers: int = MAX_WORKERS,
) -> dict:
    """Evaluate on MATH benchmark (accuracy)."""
    reset_cost_tracking()
    details = []

    if parallel and len(samples) > 1:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(samples))) as executor:
            futures = {
                executor.submit(_eval_single_math_bench, agent_fn, s, i): i
                for i, s in enumerate(samples)
            }
            for future in as_completed(futures):
                details.append(future.result())
    else:
        for i, s in enumerate(samples):
            details.append(_eval_single_math_bench(agent_fn, s, i))

    details.sort(key=lambda x: x["idx"])
    correct = sum(1 for d in details if d.get("correct", False))
    total = len(samples)

    return {
        "benchmark": "math",
        "score": round(correct / total * 100, 2) if total > 0 else 0.0,
        "correct": correct,
        "total": total,
        "cost_usd": round(get_session_cost(), 4),
        "details": details,
    }


# ─── HumanEval ────────────────────────────────────

def load_humaneval(n: Optional[int] = None, seed: int = 42) -> list[dict]:
    """Load HumanEval benchmark."""
    ds = load_dataset("openai/openai_humaneval", split="test")
    samples = []
    for item in ds:
        samples.append({
            "task_id": item["task_id"],
            "prompt": item["prompt"],
            "test": item["test"],
            "entry_point": item["entry_point"],
            "canonical_solution": item["canonical_solution"],
        })
    if n is not None and n < len(samples):
        rng = random.Random(seed)
        samples = rng.sample(samples, n)
    return samples


def _eval_single_humaneval(agent_fn, sample, idx):
    """Evaluate a single HumanEval sample."""
    try:
        response = agent_fn(sample["prompt"])
        # Extract code from response
        code_match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
        if code_match:
            code = code_match.group(1)
        else:
            code = response

        # Build complete program
        full_code = sample["prompt"] + code + "\n" + sample["test"] + f"\ncheck({sample['entry_point']})"

        # Execute safely
        import io, contextlib
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                exec(full_code, {})
            return {"idx": idx, "correct": True, "task_id": sample["task_id"]}
        except Exception as e:
            return {"idx": idx, "correct": False, "task_id": sample["task_id"], "error": str(e)[:100]}

    except Exception as e:
        return {"idx": idx, "correct": False, "error": str(e)[:100]}


def evaluate_humaneval(
    agent_fn: Callable[[str], str],
    samples: list[dict],
    parallel: bool = False,  # Code execution not thread-safe
    max_workers: int = 4,
) -> dict:
    """Evaluate on HumanEval (pass@1)."""
    reset_cost_tracking()
    details = []

    # Run sequentially for safety (code execution)
    for i, s in enumerate(samples):
        details.append(_eval_single_humaneval(agent_fn, s, i))

    correct = sum(1 for d in details if d.get("correct", False))
    total = len(samples)

    return {
        "benchmark": "humaneval",
        "score": round(correct / total * 100, 2) if total > 0 else 0.0,
        "correct": correct,
        "total": total,
        "cost_usd": round(get_session_cost(), 4),
        "details": details,
    }
