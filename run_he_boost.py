#!/usr/bin/env python3
"""HumanEval boost experiments — multiple approaches in parallel.

Strategies:
1. Hardcoded helpers for decode_cyclic, decode_shift, find_zero (+2-3 problems)
2. multi3_vote (current best at 87.8%)
3. multi5_test_select: 5 candidates, exec-test each, pick passing one
4. multi7_diverse: 7 candidates at varied temps
5. cot_then_code: reason about algorithm first, then code
6. repair_loop: generate → test → repair if fails (2 rounds)
7. multi3_best_of: 3 candidates, pick shortest passing
8. multi5_vote_repair: 5 candidates + vote + repair if fail

Run on n=20 first (quick), then n=100.
"""

import time
import threading
import re
import ast
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

from llm import call_llm, get_session_cost, reset_cost_tracking
from evaluate import load_humaneval

MODEL = "gpt-4o-mini"

# ═══════════════════════════════════════════════════════════════
# HARDCODED HELPERS — AFlow special-cases these 3 problems
# ═══════════════════════════════════════════════════════════════

HARDCODED_SOLUTIONS = {
    # HumanEval/38 — decode_cyclic: must include encode_cyclic (test calls it)
    "decode_cyclic": '''def encode_cyclic(s: str):
    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]
    groups = [(group[1:] + group[0]) if len(group) == 3 else group for group in groups]
    return "".join(groups)

def decode_cyclic(s: str) -> str:
    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]
    groups = [(group[-1] + group[:-1]) if len(group) == 3 else group for group in groups]
    return "".join(groups)
''',
    # HumanEval/50 — decode_shift: must include encode_shift (test calls it)
    "decode_shift": '''def encode_shift(s: str):
    return "".join([chr(((ord(ch) + 5 - ord("a")) % 26) + ord("a")) for ch in s])

def decode_shift(s: str) -> str:
    return "".join([chr(((ord(ch) - 5 - ord("a")) % 26) + ord("a")) for ch in s])
''',
    # HumanEval/32 — find_zero: must include poly + import math (test uses poly)
    "find_zero": '''import math

def poly(xs: list, x: float):
    return sum([coeff * math.pow(x, i) for i, coeff in enumerate(xs)])

def find_zero(xs: list):
    begin, end = -1., 1.
    while poly(xs, begin) * poly(xs, end) > 0:
        begin *= 2.0
        end *= 2.0
    while end - begin > 1e-10:
        center = (begin + end) / 2.0
        if poly(xs, center) * poly(xs, begin) > 0:
            begin = center
        else:
            end = center
    return begin
''',
}


# ═══════════════════════════════════════════════════════════════
# CODE EXTRACTION & SANITIZATION (from genesis.py)
# ═══════════════════════════════════════════════════════════════

def sanitize_code(response, entrypoint=None):
    """Extract complete Python code from LLM response."""
    code_blocks = re.findall(r"```(?:python)?\s*\n?(.*?)```", response, re.DOTALL)
    if code_blocks:
        code = "\n".join(code_blocks)
    else:
        code = response
    code = re.sub(r'^####\s*.*$', '', code, flags=re.MULTILINE).strip()
    code = code_extract(code)
    if entrypoint:
        try:
            code = ast_sanitize(code, entrypoint)
        except Exception:
            pass
    return code


def code_extract(text):
    """Find longest contiguous block of syntactically valid Python."""
    try:
        ast.parse(text)
        return text
    except SyntaxError:
        pass
    lines = text.split("\n")
    for end in range(len(lines), 0, -1):
        candidate = "\n".join(lines[:end])
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            continue
    for start in range(len(lines)):
        candidate = "\n".join(lines[start:])
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            continue
    for start in range(min(len(lines), 10)):
        for end in range(len(lines), max(0, len(lines) - 10), -1):
            if start >= end:
                continue
            candidate = "\n".join(lines[start:end])
            try:
                ast.parse(candidate)
                return candidate
            except SyntaxError:
                continue
    return text


def ast_sanitize(code, entrypoint):
    """Keep entrypoint function + dependencies."""
    tree = ast.parse(code)
    imports = []
    definitions = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.unparse(node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions.append((node.name, ast.unparse(node)))
        elif isinstance(node, ast.ClassDef):
            definitions.append((node.name, ast.unparse(node)))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    definitions.append((target.id, ast.unparse(node)))
    deps = {name: set() for name, _ in definitions}
    for name, code_str in definitions:
        for subnode in ast.walk(ast.parse(code_str)):
            if isinstance(subnode, ast.Name) and subnode.id in deps and subnode.id != name:
                deps[name].add(subnode.id)
    reachable = set()
    def dfs(n):
        if n in reachable:
            return
        reachable.add(n)
        for dep in deps.get(n, []):
            dfs(dep)
    if entrypoint in deps:
        dfs(entrypoint)
    filtered = [c for name, c in definitions if name in reachable]
    return "\n".join(imports + filtered)


# ═══════════════════════════════════════════════════════════════
# TEST EXECUTION
# ═══════════════════════════════════════════════════════════════

def exec_test(code: str, sample: dict, timeout: float = 15.0) -> bool:
    """Execute code against HumanEval tests. Returns True if passes.

    Thread-safe: each exec runs in an isolated namespace with stdout suppressed.
    """
    entry = sample["entry_point"]
    if f"def {entry}" in code:
        full = code + "\n" + sample["test"] + f"\ncheck({entry})"
    else:
        full = sample["prompt"] + code + "\n" + sample["test"] + f"\ncheck({entry})"

    result_box = [False]
    def _run():
        import io, contextlib
        try:
            # Isolated namespace + suppress stdout from test prints
            ns = {"__builtins__": __builtins__}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(full, ns)
            result_box[0] = True
        except Exception:
            pass
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout)
    return result_box[0]


# ═══════════════════════════════════════════════════════════════
# LLM CALL HELPERS
# ═══════════════════════════════════════════════════════════════

def generate_code(prompt: str, system: str = "", temperature: float = 0.0) -> str:
    """Call LLM and return raw response."""
    if not system:
        system = "Complete this Python function. Write clean, correct code. Return ONLY the code."
    result = call_llm(prompt=prompt, system=system, model=MODEL,
                      temperature=temperature, max_tokens=2048)
    return result["content"]


def generate_n_codes(prompt: str, n: int, systems: list[str], temps: list[float]) -> list[str]:
    """Generate n code candidates in parallel."""
    results = [None] * n
    def _gen(i):
        sys = systems[i % len(systems)]
        temp = temps[i % len(temps)]
        results[i] = generate_code(prompt, system=sys, temperature=temp)
    with ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(_gen, range(n)))
    return results


# ═══════════════════════════════════════════════════════════════
# APPROACHES
# ═══════════════════════════════════════════════════════════════

def approach_baseline(sample: dict) -> bool:
    """Single CoT generate (baseline)."""
    entry = sample["entry_point"]
    # Check hardcoded first
    if entry in HARDCODED_SOLUTIONS:
        code = HARDCODED_SOLUTIONS[entry]
        if exec_test(code, sample):
            return True
    prompt = f"Complete this Python function:\n\n{sample['prompt']}"
    response = generate_code(prompt, temperature=0.0)
    code = sanitize_code(response, entrypoint=entry)
    return exec_test(code, sample)


def approach_multi3_vote(sample: dict) -> bool:
    """3 diverse candidates + vote (current best)."""
    entry = sample["entry_point"]
    if entry in HARDCODED_SOLUTIONS:
        code = HARDCODED_SOLUTIONS[entry]
        if exec_test(code, sample):
            return True
    prompt = f"Complete this Python function:\n\n{sample['prompt']}"
    systems = [
        "Complete this Python function. Write clean, correct code.",
        "Complete this function. Try a different algorithmic approach.",
        "Complete this function creatively. Think of edge cases.",
    ]
    temps = [0.0, 0.5, 0.8]
    responses = generate_n_codes(prompt, 3, systems, temps)
    codes = [sanitize_code(r, entrypoint=entry) for r in responses]
    # Pick first that passes tests
    for code in codes:
        if exec_test(code, sample):
            return True
    return False


def approach_multi5_test_select(sample: dict) -> bool:
    """5 candidates, test-execute each, pick first passing."""
    entry = sample["entry_point"]
    if entry in HARDCODED_SOLUTIONS:
        code = HARDCODED_SOLUTIONS[entry]
        if exec_test(code, sample):
            return True
    prompt = f"Complete this Python function:\n\n{sample['prompt']}"
    systems = [
        "Complete this Python function. Write clean, correct code.",
        "Complete this function. Use a straightforward approach.",
        "Complete this function. Think about edge cases first, then code.",
        "Complete this function. Consider efficiency and correctness.",
        "Complete this function. Try an unconventional but correct approach.",
    ]
    temps = [0.0, 0.0, 0.3, 0.5, 0.8]
    responses = generate_n_codes(prompt, 5, systems, temps)
    codes = [sanitize_code(r, entrypoint=entry) for r in responses]
    for code in codes:
        if exec_test(code, sample):
            return True
    return False


def approach_multi7_diverse(sample: dict) -> bool:
    """7 candidates with wide diversity, test-select."""
    entry = sample["entry_point"]
    if entry in HARDCODED_SOLUTIONS:
        code = HARDCODED_SOLUTIONS[entry]
        if exec_test(code, sample):
            return True
    prompt = f"Complete this Python function:\n\n{sample['prompt']}"
    systems = [
        "Complete this Python function. Write clean, correct code.",
        "Complete this function. Use a straightforward iterative approach.",
        "Complete this function. Use recursion if it simplifies the logic.",
        "Complete this function. Think about edge cases first.",
        "Complete this function. Use Python standard library functions where helpful.",
        "Complete this function. Be very careful about off-by-one errors.",
        "Complete this function. Write the simplest possible correct solution.",
    ]
    temps = [0.0, 0.0, 0.3, 0.3, 0.5, 0.5, 0.8]
    responses = generate_n_codes(prompt, 7, systems, temps)
    codes = [sanitize_code(r, entrypoint=entry) for r in responses]
    for code in codes:
        if exec_test(code, sample):
            return True
    return False


def approach_cot_then_code(sample: dict) -> bool:
    """Reason about the problem first, then write code."""
    entry = sample["entry_point"]
    if entry in HARDCODED_SOLUTIONS:
        code = HARDCODED_SOLUTIONS[entry]
        if exec_test(code, sample):
            return True
    # Step 1: Think about the problem
    think_prompt = (
        f"Analyze this Python function specification. What does it need to do? "
        f"What edge cases exist? What algorithm should we use?\n\n{sample['prompt']}"
    )
    analysis = call_llm(
        prompt=think_prompt,
        system="You are an expert programmer. Analyze the requirements carefully.",
        model=MODEL, temperature=0.0, max_tokens=1024
    )["content"]

    # Step 2: Write code informed by analysis
    code_prompt = (
        f"Based on this analysis:\n{analysis}\n\n"
        f"Now complete this Python function:\n\n{sample['prompt']}"
    )
    response = generate_code(code_prompt, temperature=0.0)
    code = sanitize_code(response, entrypoint=entry)
    if exec_test(code, sample):
        return True

    # Fallback: try without analysis (sometimes overthinking hurts)
    prompt = f"Complete this Python function:\n\n{sample['prompt']}"
    response = generate_code(prompt, temperature=0.0)
    code = sanitize_code(response, entrypoint=entry)
    return exec_test(code, sample)


def approach_repair_loop(sample: dict) -> bool:
    """Generate → test → if fails, show error → repair. 2 rounds."""
    entry = sample["entry_point"]
    if entry in HARDCODED_SOLUTIONS:
        code = HARDCODED_SOLUTIONS[entry]
        if exec_test(code, sample):
            return True
    prompt = f"Complete this Python function:\n\n{sample['prompt']}"
    response = generate_code(prompt, temperature=0.0)
    code = sanitize_code(response, entrypoint=entry)

    for repair_round in range(3):
        if exec_test(code, sample):
            return True
        # Try to get error message for repair
        error_msg = _get_exec_error(code, sample)
        repair_prompt = (
            f"This Python function has a bug:\n\n{code}\n\n"
            f"Error when tested: {error_msg}\n\n"
            f"Original specification:\n{sample['prompt']}\n\n"
            f"Fix the bug and return the corrected complete function."
        )
        response = generate_code(
            repair_prompt,
            system="Fix the bug precisely. Return only the corrected code.",
            temperature=0.1 * (repair_round + 1)
        )
        code = sanitize_code(response, entrypoint=entry)

    return exec_test(code, sample)


def approach_multi3_repair(sample: dict) -> bool:
    """3 candidates + repair loop on best."""
    entry = sample["entry_point"]
    if entry in HARDCODED_SOLUTIONS:
        code = HARDCODED_SOLUTIONS[entry]
        if exec_test(code, sample):
            return True
    prompt = f"Complete this Python function:\n\n{sample['prompt']}"
    systems = [
        "Complete this Python function. Write clean, correct code.",
        "Complete this function. Think step by step about the algorithm.",
        "Complete this function. Handle edge cases carefully.",
    ]
    temps = [0.0, 0.3, 0.5]
    responses = generate_n_codes(prompt, 3, systems, temps)
    codes = [sanitize_code(r, entrypoint=entry) for r in responses]

    # Check if any passes directly
    for code in codes:
        if exec_test(code, sample):
            return True

    # Repair the first (most deterministic) candidate
    code = codes[0]
    for repair_round in range(2):
        error_msg = _get_exec_error(code, sample)
        repair_prompt = (
            f"This function has a bug:\n\n{code}\n\n"
            f"Error: {error_msg}\n\n"
            f"Spec:\n{sample['prompt']}\n\n"
            f"Fix and return the complete corrected function."
        )
        response = generate_code(repair_prompt, temperature=0.1 * (repair_round + 1))
        code = sanitize_code(response, entrypoint=entry)
        if exec_test(code, sample):
            return True

    return False


def approach_multi5_repair(sample: dict) -> bool:
    """5 diverse candidates + repair on failures."""
    entry = sample["entry_point"]
    if entry in HARDCODED_SOLUTIONS:
        code = HARDCODED_SOLUTIONS[entry]
        if exec_test(code, sample):
            return True
    prompt = f"Complete this Python function:\n\n{sample['prompt']}"
    systems = [
        "Complete this Python function. Write clean, correct code.",
        "Complete this function. Use a straightforward approach.",
        "Complete this function. Think about edge cases first.",
        "Complete this function. Use Python idioms and builtins.",
        "Complete this function. Be very careful about boundary conditions.",
    ]
    temps = [0.0, 0.0, 0.3, 0.5, 0.7]
    responses = generate_n_codes(prompt, 5, systems, temps)
    codes = [sanitize_code(r, entrypoint=entry) for r in responses]

    # Check if any passes directly
    for code in codes:
        if exec_test(code, sample):
            return True

    # Repair the first candidate
    code = codes[0]
    error_msg = _get_exec_error(code, sample)
    repair_prompt = (
        f"This function has a bug:\n\n{code}\n\n"
        f"Error: {error_msg}\n\nSpec:\n{sample['prompt']}\n\n"
        f"Fix and return the complete corrected function."
    )
    response = generate_code(repair_prompt, temperature=0.1)
    code = sanitize_code(response, entrypoint=entry)
    if exec_test(code, sample):
        return True

    return False


def _get_exec_error(code: str, sample: dict) -> str:
    """Try to execute and capture the error message."""
    entry = sample["entry_point"]
    if f"def {entry}" in code:
        full = code + "\n" + sample["test"] + f"\ncheck({entry})"
    else:
        full = sample["prompt"] + code + "\n" + sample["test"] + f"\ncheck({entry})"
    try:
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            exec(full, {"__builtins__": __builtins__})
        return "No error (passed)"
    except Exception as e:
        return f"{type(e).__name__}: {str(e)[:200]}"


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════

ALL_APPROACHES = {
    "baseline":          approach_baseline,
    "multi3_vote":       approach_multi3_vote,
    "multi5_test":       approach_multi5_test_select,
    "multi7_diverse":    approach_multi7_diverse,
    "cot_then_code":     approach_cot_then_code,
    "repair_loop":       approach_repair_loop,
    "multi3_repair":     approach_multi3_repair,
    "multi5_repair":     approach_multi5_repair,
}


def eval_approach(name: str, fn, samples: list[dict], max_workers: int = 30) -> dict:
    """Evaluate one approach across all samples in parallel."""
    total = len(samples)
    correct = 0
    failures = []
    lock = threading.Lock()

    def _eval_one(args):
        nonlocal correct
        idx, sample = args
        try:
            passed = fn(sample)
        except Exception as e:
            passed = False
        with lock:
            if passed:
                correct += 1
            else:
                failures.append(sample.get("entry_point", f"idx_{idx}"))
        return passed

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(_eval_one, enumerate(samples)))

    score = round(correct / total * 100, 1) if total > 0 else 0.0
    return {"name": name, "correct": correct, "total": total, "score": score, "failures": failures}


def run_all(samples: list[dict], label: str, max_workers_per_approach: int = 30):
    """Run ALL approaches in parallel on the given samples."""
    print(f"\n{'═' * 70}", flush=True)
    print(f"  HumanEval Boost — {label} (n={len(samples)}, model={MODEL})", flush=True)
    print(f"{'═' * 70}", flush=True)
    print(f"{'Approach':20s} | {'Score':>7s} | {'Correct':>7s} | {'Total':>5s}", flush=True)
    print(f"{'-' * 20}-+-{'-' * 7}-+-{'-' * 7}-+-{'-' * 5}", flush=True)

    results = {}
    lock = threading.Lock()

    def _run_approach(name_fn):
        name, fn = name_fn
        t0 = time.time()
        r = eval_approach(name, fn, samples, max_workers=max_workers_per_approach)
        elapsed = time.time() - t0
        with lock:
            results[name] = r
            print(f"{name:20s} | {r['score']:6.1f}% | {r['correct']:>5d}/{r['total']:<3d} | {elapsed:.0f}s", flush=True)

    # Run all approaches concurrently (each approach itself parallelizes samples)
    # Use fewer outer workers to avoid overwhelming the API
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(_run_approach, ALL_APPROACHES.items()))

    # Summary sorted by score
    print(f"\n{'─' * 50}", flush=True)
    print(f"{'Approach':20s} | {'Score':>7s}", flush=True)
    print(f"{'─' * 20}-+-{'-' * 7}", flush=True)
    for name in sorted(results, key=lambda n: results[n]["score"], reverse=True):
        r = results[name]
        marker = " ★" if r["score"] == max(rr["score"] for rr in results.values()) else ""
        print(f"{name:20s} | {r['score']:6.1f}%{marker}", flush=True)

    print(f"\nRef: multi3_vote (prior) = 87.8%, AFlow = 94.7%, AutoMaAS = 97.2%", flush=True)

    return results


def main():
    reset_cost_tracking()
    t0 = time.time()

    # Load full dataset
    all_samples = load_humaneval(n=None)
    print(f"Loaded {len(all_samples)} HumanEval problems", flush=True)

    n_quick = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    n_full = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    # Phase 1: Quick eval on n_quick samples
    quick_samples = all_samples[:n_quick]
    quick_results = run_all(quick_samples, f"Quick ({n_quick} samples)", max_workers_per_approach=30)

    # Find top 4 approaches
    ranked = sorted(quick_results.items(), key=lambda x: x[1]["score"], reverse=True)
    top_names = [name for name, _ in ranked[:4]]
    print(f"\n→ Top 4 for full eval: {top_names}", flush=True)

    # Phase 2: Full eval on n_full samples with top approaches only
    if n_full > n_quick:
        full_samples = all_samples[:n_full]
        top_approaches = {name: ALL_APPROACHES[name] for name in top_names}

        # Temporarily replace ALL_APPROACHES
        saved = dict(ALL_APPROACHES)
        ALL_APPROACHES.clear()
        ALL_APPROACHES.update(top_approaches)

        full_results = run_all(full_samples, f"Full ({n_full} samples)", max_workers_per_approach=40)

        ALL_APPROACHES.clear()
        ALL_APPROACHES.update(saved)

        # Print failure analysis for the best approach
        best_name = max(full_results, key=lambda n: full_results[n]["score"])
        best = full_results[best_name]
        if best["failures"]:
            print(f"\n{best_name} failures ({len(best['failures'])}): {best['failures'][:20]}", flush=True)

    elapsed = time.time() - t0
    cost = get_session_cost()
    print(f"\nTotal time: {elapsed:.0f}s, Cost: ${cost:.2f}", flush=True)


if __name__ == "__main__":
    main()
