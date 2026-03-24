#!/usr/bin/env python3
"""Push MATH score: diverse candidates + verification + code assist."""

import time
import re
from collections import Counter
from evaluate import load_math, evaluate_math_bench, extract_math_answer, normalize_math_answer
from llm import call_llm, get_session_cost, reset_cost_tracking

MODEL = "gpt-4o-mini"


def solve_math_combined(problem, n_candidates=3, verify=True):
    """Multi-candidate math solving with verification."""
    answers = []
    responses = []

    # Generate diverse candidates
    temps = [0.0, 0.3, 0.6, 0.8, 0.5][:n_candidates]
    for t in temps:
        r = call_llm(
            prompt=f"Problem: {problem}",
            system=(
                "You are a mathematics professor. Solve step by step with full rigor. "
                "Put your final answer in \\boxed{...} format."
            ),
            model=MODEL, temperature=t, max_tokens=2048,
        )
        content = r["content"]
        responses.append(content)
        ans = extract_math_answer(content)
        if ans:
            answers.append(normalize_math_answer(ans))

    if not answers:
        return responses[0] if responses else ""

    # If all agree, return
    counter = Counter(answers)
    most_common, count = counter.most_common(1)[0]
    if count >= 2:
        return f"\\boxed{{{most_common}}}"

    # If no consensus, verify with code
    if verify:
        code_r = call_llm(
            prompt=(
                f"Solve this math problem using Python/sympy code. "
                f"Print ONLY the final answer.\n\nProblem: {problem}"
            ),
            system="Expert mathematician using Python. Print only the answer.",
            model=MODEL, temperature=0.0, max_tokens=2048,
        )
        code_content = code_r["content"]
        code_match = re.search(r'```python\s*(.*?)\s*```', code_content, re.DOTALL)
        if code_match:
            code = code_match.group(1)
            try:
                import io, contextlib, sympy
                stdout = io.StringIO()
                ns = {'sympy': sympy, 'print': print, 'range': range, 'int': int,
                      'float': float, 'abs': abs, 'min': min, 'max': max, 'round': round}
                for name in ['sqrt', 'Rational', 'simplify', 'solve', 'symbols', 'Symbol',
                             'factor', 'expand', 'pi', 'gcd', 'lcm', 'binomial', 'factorial',
                             'ceiling', 'floor', 'Mod', 'isprime', 'log', 'exp', 'sin', 'cos',
                             'tan', 'oo', 'E']:
                    if hasattr(sympy, name):
                        ns[name] = getattr(sympy, name)
                import math
                ns['math'] = math
                with contextlib.redirect_stdout(stdout):
                    exec(code, ns)
                output = stdout.getvalue().strip()
                if output:
                    code_ans = normalize_math_answer(output.split('\n')[-1])
                    answers.append(code_ans)
            except Exception:
                pass

    # Final vote
    counter = Counter(answers)
    return f"\\boxed{{{counter.most_common(1)[0][0]}}}"


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
    parser.add_argument("--n", type=int, default=500)
    args = parser.parse_args()

    samples = load_math(n=args.n, seed=42)
    print(f"=== MATH Combined (gpt-4o-mini, n={len(samples)}) ===")
    print(f"Published: MaAS=51.82%")
    print()
    print(f"{'Method':25s} | {'Score':8s} | {'Cost':8s} | {'Time':5s} | Correct")
    print("-" * 65)

    # Strategy 1: 3 candidates + verification
    run_math("3cand_verify", lambda p: solve_math_combined(p, 3, True), samples)

    # Strategy 2: 5 candidates + verification
    run_math("5cand_verify", lambda p: solve_math_combined(p, 5, True), samples)


if __name__ == "__main__":
    main()
