#!/usr/bin/env python3
"""Push MATH score above MaAS's 51.82% — try code-based solving + multi-attempt."""

import json
import time
import re
from evaluate import load_math, evaluate_math_bench
from llm import call_llm, get_session_cost, reset_cost_tracking

MODEL = "gpt-4o-mini"


def math_code_solve(problem):
    """Solve MATH problems by generating Python code."""
    result = call_llm(
        prompt=(
            f"Solve this math problem by writing Python code that computes the answer.\n\n"
            f"Problem: {problem}\n\n"
            f"Write Python code that prints ONLY the final answer. "
            f"Use sympy for symbolic computation if needed. "
            f"The answer should be in simplest form (fraction, integer, or expression)."
        ),
        system="You are an expert mathematician who solves problems using Python/sympy. Write clean code.",
        model=MODEL, temperature=0.0, max_tokens=2048,
    )

    content = result["content"]
    code_match = re.search(r'```python\s*(.*?)\s*```', content, re.DOTALL)
    code = code_match.group(1) if code_match else content

    try:
        import io, contextlib
        stdout = io.StringIO()
        import sympy
        namespace = {'sympy': sympy, 'print': print, 'range': range, 'int': int, 'float': float,
                     'str': str, 'abs': abs, 'min': min, 'max': max, 'sum': sum, 'len': len,
                     'list': list, 'dict': dict, 'set': set, 'tuple': tuple, 'sorted': sorted,
                     'enumerate': enumerate, 'zip': zip, 'round': round, 'pow': pow,
                     'True': True, 'False': False, 'None': None}
        # Add common sympy imports
        for name in ['sqrt', 'Rational', 'simplify', 'solve', 'symbols', 'Symbol',
                     'factor', 'expand', 'pi', 'E', 'oo', 'sin', 'cos', 'tan',
                     'log', 'exp', 'Eq', 'latex', 'gcd', 'lcm', 'binomial',
                     'factorial', 'ceiling', 'floor', 'Mod', 'isprime']:
            if hasattr(sympy, name):
                namespace[name] = getattr(sympy, name)
        import math
        namespace['math'] = math

        with contextlib.redirect_stdout(stdout):
            exec(code, namespace)
        output = stdout.getvalue().strip()
        if output:
            return f"\\boxed{{{output.split(chr(10))[-1]}}}"
    except Exception:
        pass

    # Fallback to CoT
    return math_cot(problem)


def math_cot(problem):
    """Solve MATH with strong CoT prompt."""
    result = call_llm(
        prompt=f"Problem: {problem}",
        system=(
            "You are a mathematics professor. Solve this problem step by step with full rigor. "
            "Put your final answer in \\boxed{...} format."
        ),
        model=MODEL, temperature=0.0, max_tokens=2048,
    )
    return result["content"]


def math_cot_refine(problem):
    """CoT + self-refinement for MATH."""
    # Solve
    solution = call_llm(
        prompt=f"Problem: {problem}",
        system="Mathematics professor. Solve step by step. Final answer in \\boxed{{}}.",
        model=MODEL, temperature=0.0, max_tokens=2048,
    )

    # Verify
    verify = call_llm(
        prompt=(
            f"Problem: {problem}\n\n"
            f"Proposed solution:\n{solution['content']}\n\n"
            f"Carefully verify this solution. Check every calculation step. "
            f"If you find errors, provide the corrected solution with answer in \\boxed{{}}. "
            f"If correct, confirm with the answer in \\boxed{{}}."
        ),
        system="Meticulous math verifier. Check every step.",
        model=MODEL, temperature=0.0, max_tokens=2048,
    )
    return verify["content"]


def math_ensemble(problem):
    """Solve with multiple methods, pick best answer."""
    from collections import Counter

    answers = []
    responses = []

    # Method 1: CoT
    r1 = math_cot(problem)
    responses.append(r1)

    # Method 2: Code solve
    r2 = math_code_solve(problem)
    responses.append(r2)

    # Extract answers
    for r in responses:
        match = re.search(r'\\boxed\{([^}]+)\}', r)
        if match:
            answers.append(match.group(1).strip())

    if not answers:
        return r1

    # If both agree, great
    if len(answers) >= 2 and answers[0] == answers[1]:
        return f"\\boxed{{{answers[0]}}}"

    # If they disagree, do a tiebreaker
    if len(set(answers)) > 1:
        r3 = math_cot_refine(problem)
        match = re.search(r'\\boxed\{([^}]+)\}', r3)
        if match:
            answers.append(match.group(1).strip())

    counter = Counter(answers)
    best = counter.most_common(1)[0][0]
    return f"\\boxed{{{best}}}"


def run_math(name, agent_fn, samples):
    reset_cost_tracking()
    start = time.time()
    result = evaluate_math_bench(agent_fn, samples)
    elapsed = time.time() - start
    cost = get_session_cost()
    print(f"  {name:25s} | {result['score']:6.2f}% | ${cost:.4f} | {elapsed:.0f}s | {result['correct']}/{result['total']}")
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()

    samples = load_math(n=args.n, seed=42)
    print(f"=== MATH Push (gpt-4o-mini, n={len(samples)}) ===")
    print(f"Published: MaAS=51.82%, AFlow=51.3%")
    print()
    print(f"{'Method':25s} | {'Score':8s} | {'Cost':8s} | {'Time':5s} | Correct")
    print("-" * 65)

    # Install sympy for code-based solving
    import subprocess
    subprocess.run(["pip", "install", "sympy", "-q"], capture_output=True)

    run_math("cot", math_cot, samples)
    run_math("cot_refine", math_cot_refine, samples)
    run_math("code_solve", math_code_solve, samples)
    run_math("ensemble", math_ensemble, samples)


if __name__ == "__main__":
    main()
