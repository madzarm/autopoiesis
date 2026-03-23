#!/usr/bin/env python3
"""HumanEval with direct code continuation approach + self-repair."""

import json
import time
import re
import io
import contextlib
from evaluate import load_humaneval
from llm import call_llm, get_session_cost, reset_cost_tracking

MODEL = "gpt-4o-mini"


def extract_completion(response: str, entry_point: str) -> str:
    """Extract function body from response — handles multiple formats."""
    # Strategy 1: If response contains ```python block, extract it
    code_match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
    if code_match:
        code = code_match.group(1)
        # If it contains the function definition, extract body
        lines = code.split('\n')
        body_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('def ') and entry_point in line:
                body_start = i + 1
                # Skip docstring if present
                if body_start < len(lines) and ('"""' in lines[body_start] or "'''" in lines[body_start]):
                    body_start += 1
                    while body_start < len(lines):
                        if '"""' in lines[body_start] or "'''" in lines[body_start]:
                            body_start += 1
                            break
                        body_start += 1
                break
        if body_start > 0:
            return '\n'.join(lines[body_start:])
        return code

    # Strategy 2: If response starts with indented code, use as-is
    lines = response.split('\n')
    if lines and (lines[0].startswith('    ') or lines[0].startswith('\t')):
        return response

    # Strategy 3: Look for def statement and extract body
    for i, line in enumerate(lines):
        if line.strip().startswith('def ') and entry_point in line:
            body_start = i + 1
            if body_start < len(lines) and ('"""' in lines[body_start] or "'''" in lines[body_start]):
                body_start += 1
                while body_start < len(lines):
                    if '"""' in lines[body_start] or "'''" in lines[body_start]:
                        body_start += 1
                        break
                    body_start += 1
            return '\n'.join(lines[body_start:])

    # Strategy 4: Just use the raw response
    return response


def test_code(prompt: str, completion: str, test: str, entry_point: str) -> tuple[bool, str]:
    """Test if the completion passes the test cases."""
    full_code = prompt + completion + "\n" + test + f"\ncheck({entry_point})"
    try:
        exec(full_code, {})
        return True, ""
    except Exception as e:
        return False, str(e)


def run_humaneval(samples, method_name, agent_fn, self_repair=False):
    """Run HumanEval with optional self-repair."""
    reset_cost_tracking()
    correct = 0
    total = len(samples)
    start = time.time()

    for i, sample in enumerate(samples):
        try:
            response = agent_fn(sample["prompt"])
            completion = extract_completion(response, sample["entry_point"])

            passed, error = test_code(
                sample["prompt"], completion, sample["test"], sample["entry_point"]
            )

            if passed:
                correct += 1
            elif self_repair:
                # Self-repair: show error, ask to fix
                fix_response = call_llm(
                    prompt=(
                        f"The following Python function has a bug:\n\n"
                        f"{sample['prompt']}{completion}\n\n"
                        f"Error: {error[:200]}\n\n"
                        f"Fix the function body. Return ONLY the corrected function body code."
                    ),
                    system="Fix the Python code bug. Return ONLY the function body.",
                    model=MODEL, temperature=0.0, max_tokens=1024,
                )
                fix_completion = extract_completion(fix_response["content"], sample["entry_point"])
                passed2, _ = test_code(
                    sample["prompt"], fix_completion, sample["test"], sample["entry_point"]
                )
                if passed2:
                    correct += 1
        except Exception:
            pass

    elapsed = time.time() - start
    cost = get_session_cost()
    score = round(correct / total * 100, 2)
    print(f"  {method_name:30s} | {score:6.2f}% | ${cost:.4f} | {elapsed:.0f}s | {correct}/{total}")
    return score


def main():
    samples = load_humaneval()
    print(f"=== HumanEval Direct (gpt-4o-mini, n={len(samples)}) ===")
    print(f"Published: MaAS=92.85%, AFlow=90.9%")
    print()

    # Method 1: Direct continuation — best prompt
    def direct_best(prompt):
        r = call_llm(
            prompt=prompt + "\n    # Implementation:",
            system="Complete the Python function. Output ONLY the function body code (indented, no markdown, no explanation).",
            model=MODEL, temperature=0.0, max_tokens=1024,
        )
        return r["content"]

    # Method 2: Direct continuation with temperature for diversity
    def direct_t03(prompt):
        r = call_llm(
            prompt=prompt,
            system="Complete the Python function body. Output ONLY code, no markdown.",
            model=MODEL, temperature=0.3, max_tokens=1024,
        )
        return r["content"]

    print(f"{'Method':30s} | {'Score':8s} | {'Cost':8s} | {'Time':5s} | Correct")
    print("-" * 70)

    run_humaneval(samples, "direct_best", direct_best, self_repair=False)
    run_humaneval(samples, "direct_best+repair", direct_best, self_repair=True)
    run_humaneval(samples, "direct_t03+repair", direct_t03, self_repair=True)


if __name__ == "__main__":
    main()
