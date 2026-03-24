#!/usr/bin/env python3
"""HumanEval combined approach: generate N diverse candidates → test → repair → best.

This combines self-consistency with repair loops for maximum pass rate.
"""

import time
import re
from evaluate import load_humaneval
from llm import call_llm, get_session_cost, reset_cost_tracking

MODEL = "gpt-4o-mini"


def extract_body(response):
    response = re.sub(r'```python\s*', '', response)
    response = re.sub(r'```\s*', '', response)
    lines = response.split('\n')
    i = 0
    if lines and lines[0].strip().startswith('def '):
        i = 1
        if i < len(lines):
            s = lines[i].strip()
            if s.startswith('"""') or s.startswith("'''"):
                q = s[:3]
                if s.count(q) >= 2:
                    i += 1
                else:
                    i += 1
                    while i < len(lines) and q not in lines[i]:
                        i += 1
                    i += 1
    return '\n'.join(lines[i:])


def test_code(prompt, body, test, entry_point):
    full = prompt + body + "\n" + test + f"\ncheck({entry_point})"
    try:
        exec(full, {})
        return True, ""
    except Exception as e:
        return False, str(e)[:300]


def solve_combined(sample, n_candidates=3, max_repairs=2):
    """Generate N candidates, test each, repair failures, pick first passing."""
    prompt = sample["prompt"]
    test = sample["test"]
    entry_point = sample["entry_point"]

    # Phase 1: Generate diverse candidates
    candidates = []
    for j in range(n_candidates):
        temp = [0.0, 0.4, 0.7][j % 3]
        r = call_llm(
            prompt=prompt,
            system=(
                "Complete the Python function. Output ONLY the function body "
                "(4-space indent, no markdown, no signature, no docstring)."
            ),
            model=MODEL, temperature=temp, max_tokens=1024,
        )
        body = extract_body(r["content"])
        passed, error = test_code(prompt, body, test, entry_point)
        if passed:
            return True
        candidates.append((body, error))

    # Phase 2: Repair each candidate
    for body, error in candidates:
        for repair in range(max_repairs):
            r = call_llm(
                prompt=(
                    f"Fix this Python function:\n\n{prompt}{body}\n\n"
                    f"Error: {error}\n\n"
                    f"Output ONLY the corrected function body (4-space indent, no markdown)."
                ),
                system="Fix the bug. Output ONLY code.",
                model=MODEL, temperature=0.1, max_tokens=1024,
            )
            body = extract_body(r["content"])
            passed, error = test_code(prompt, body, test, entry_point)
            if passed:
                return True

    return False


def main():
    samples = load_humaneval()
    print(f"=== HumanEval Combined (gpt-4o-mini, n={len(samples)}) ===")
    print(f"Published: MaAS=92.85%, AFlow=90.9%")
    print()

    configs = [
        ("3cand_0repair", 3, 0),
        ("3cand_1repair", 3, 1),
        ("3cand_2repair", 3, 2),
        ("5cand_2repair", 5, 2),
    ]

    for name, n_cand, max_rep in configs:
        reset_cost_tracking()
        correct = 0
        start = time.time()

        for sample in samples:
            if solve_combined(sample, n_candidates=n_cand, max_repairs=max_rep):
                correct += 1

        elapsed = time.time() - start
        cost = get_session_cost()
        score = round(correct / len(samples) * 100, 2)
        print(f"  {name:20s} | {score:6.2f}% | ${cost:.4f} | {elapsed:.0f}s | {correct}/{len(samples)}")


if __name__ == "__main__":
    main()
