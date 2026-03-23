#!/usr/bin/env python3
"""HumanEval with iterative repair loop — generate, test, fix, repeat."""

import time
import re
from evaluate import load_humaneval
from llm import call_llm, get_session_cost, reset_cost_tracking

MODEL = "gpt-4o-mini"


def extract_body(response: str) -> str:
    """Extract function body from response."""
    # Remove markdown
    response = re.sub(r'```python\s*', '', response)
    response = re.sub(r'```\s*', '', response)

    lines = response.split('\n')

    # Skip any 'def' line and docstring
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
    """Test code and return (passed, error_msg)."""
    full = prompt + body + "\n" + test + f"\ncheck({entry_point})"
    try:
        exec(full, {})
        return True, ""
    except Exception as e:
        return False, str(e)[:300]


def solve_with_repair(sample, max_repairs=3):
    """Generate code, test it, repair if needed."""
    prompt = sample["prompt"]
    test = sample["test"]
    entry_point = sample["entry_point"]

    # Initial generation
    result = call_llm(
        prompt=prompt,
        system=(
            "Complete the Python function. Output ONLY the function body code "
            "(indented with 4 spaces). No markdown, no signature, no docstring."
        ),
        model=MODEL, temperature=0.0, max_tokens=1024,
    )
    body = extract_body(result["content"])

    passed, error = test_code(prompt, body, test, entry_point)
    if passed:
        return True

    # Repair loop
    for repair_round in range(max_repairs):
        repair_result = call_llm(
            prompt=(
                f"This Python function has a bug:\n\n"
                f"{prompt}{body}\n\n"
                f"Error when running tests: {error}\n\n"
                f"Fix the function. Output ONLY the corrected function body "
                f"(indented with 4 spaces, no signature, no docstring, no markdown)."
            ),
            system="Fix the Python code. Output ONLY the function body.",
            model=MODEL, temperature=0.1 * (repair_round + 1), max_tokens=1024,
        )
        body = extract_body(repair_result["content"])
        passed, error = test_code(prompt, body, test, entry_point)
        if passed:
            return True

    return False


def main():
    samples = load_humaneval()
    print(f"=== HumanEval Repair Loop (gpt-4o-mini, n={len(samples)}) ===")
    print(f"Published: MaAS=92.85%, AFlow=90.9%")

    # Test different repair depths
    for max_repairs in [0, 1, 2, 3]:
        reset_cost_tracking()
        correct = 0
        start = time.time()

        for sample in samples:
            if max_repairs == 0:
                # No repair — just generate
                result = call_llm(
                    prompt=sample["prompt"],
                    system="Complete the Python function. Output ONLY the body (4-space indent, no markdown).",
                    model=MODEL, temperature=0.0, max_tokens=1024,
                )
                body = extract_body(result["content"])
                passed, _ = test_code(sample["prompt"], body, sample["test"], sample["entry_point"])
                if passed:
                    correct += 1
            else:
                if solve_with_repair(sample, max_repairs=max_repairs):
                    correct += 1

        elapsed = time.time() - start
        cost = get_session_cost()
        score = round(correct / len(samples) * 100, 2)
        name = f"repair_{max_repairs}"
        print(f"  {name:20s} | {score:6.2f}% | ${cost:.4f} | {elapsed:.0f}s | {correct}/{len(samples)}")


if __name__ == "__main__":
    main()
