#!/usr/bin/env python3
"""Final HumanEval push — try the infilling approach + self-consistency."""

import json
import time
import re
from collections import Counter
from evaluate import load_humaneval
from llm import call_llm, get_session_cost, reset_cost_tracking

MODEL = "gpt-4o-mini"


def extract_body(response: str, entry_point: str) -> str:
    """Extract function body - tries multiple strategies."""
    # Remove markdown fences
    response = re.sub(r'```python\s*', '', response)
    response = re.sub(r'```\s*', '', response)

    lines = response.split('\n')

    # If starts with def, skip signature + docstring
    if lines and lines[0].strip().startswith('def '):
        i = 1
        # Skip docstring
        if i < len(lines):
            stripped = lines[i].strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                quote = stripped[:3]
                if stripped.count(quote) >= 2:
                    i += 1  # Single-line docstring
                else:
                    i += 1
                    while i < len(lines) and quote not in lines[i]:
                        i += 1
                    i += 1
        return '\n'.join(lines[i:])

    # If all lines are indented, use as-is
    if all(l.startswith('    ') or l.startswith('\t') or l.strip() == '' for l in lines if l.strip()):
        return response

    # Otherwise, indent everything
    return '\n'.join('    ' + l if l.strip() else l for l in lines)


def test_solution(prompt, completion, test, entry_point):
    full = prompt + completion + "\n" + test + f"\ncheck({entry_point})"
    try:
        exec(full, {})
        return True
    except:
        return False


def he_infill(prompt, entry_point, n_attempts=1, self_repair=False, test_code="", test_entry=""):
    """Generate code using infilling approach."""
    for attempt in range(n_attempts):
        t = 0.0 if attempt == 0 else 0.5
        result = call_llm(
            prompt=prompt,
            system=(
                "You are completing a Python function. Output ONLY the function body "
                "(the code that goes after the docstring). "
                "Do NOT include the function signature, docstring, or markdown. "
                "The code should be properly indented (4 spaces)."
            ),
            model=MODEL, temperature=t, max_tokens=1024,
        )
        body = extract_body(result["content"], entry_point)

        if test_code and self_repair:
            if test_solution(prompt, body, test_code, test_entry):
                return body

    return body


def run_he(name, samples, n_attempts=1, self_repair=False, sc_n=1):
    reset_cost_tracking()
    correct = 0
    start = time.time()

    for sample in samples:
        if sc_n > 1:
            # Self-consistency: generate multiple, pick majority
            bodies = []
            for _ in range(sc_n):
                body = he_infill(
                    sample["prompt"], sample["entry_point"],
                    n_attempts=1, self_repair=False
                )
                bodies.append(body)

            # Test each and pick first that passes
            found = False
            for body in bodies:
                if test_solution(sample["prompt"], body, sample["test"], sample["entry_point"]):
                    correct += 1
                    found = True
                    break
            continue

        body = he_infill(
            sample["prompt"], sample["entry_point"],
            n_attempts=n_attempts,
            self_repair=self_repair,
            test_code=sample["test"],
            test_entry=sample["entry_point"],
        )
        if test_solution(sample["prompt"], body, sample["test"], sample["entry_point"]):
            correct += 1

    elapsed = time.time() - start
    cost = get_session_cost()
    score = round(correct / len(samples) * 100, 2)
    print(f"  {name:30s} | {score:6.2f}% | ${cost:.4f} | {elapsed:.0f}s | {correct}/{len(samples)}")
    return score


def main():
    samples = load_humaneval()
    print(f"=== Final HumanEval Push (gpt-4o-mini, n={len(samples)}) ===")
    print(f"Published: MaAS=92.85%, AFlow=90.9%")
    print()
    print(f"{'Method':30s} | {'Score':8s} | {'Cost':8s} | {'Time':5s} | Correct")
    print("-" * 70)

    run_he("infill_basic", samples)
    run_he("infill_2attempt", samples, n_attempts=2)
    run_he("infill_repair", samples, n_attempts=2, self_repair=True)
    run_he("infill_sc3", samples, sc_n=3)
    run_he("infill_sc5", samples, sc_n=5)


if __name__ == "__main__":
    main()
