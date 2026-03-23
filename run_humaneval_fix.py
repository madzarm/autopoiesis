#!/usr/bin/env python3
"""Focused HumanEval improvement — try multiple strategies."""

import json
import time
import re
import io
import contextlib
from agents_v2 import AgentV2Config, run_agent_v2
from evaluate import load_humaneval
from llm import call_llm, get_session_cost, reset_cost_tracking

MODEL = "gpt-4o-mini"


def eval_humaneval_v2(agent_fn, samples, method_name):
    """Improved HumanEval evaluation with better code extraction."""
    reset_cost_tracking()
    correct = 0
    total = len(samples)
    start = time.time()

    for i, sample in enumerate(samples):
        try:
            response = agent_fn(sample["prompt"])

            # Try multiple extraction strategies
            code = extract_code_completion(response, sample["prompt"])

            # Build and test
            full_code = sample["prompt"] + code + "\n" + sample["test"] + f"\ncheck({sample['entry_point']})"

            try:
                exec(full_code, {})
                correct += 1
            except Exception:
                pass
        except Exception:
            pass

    elapsed = time.time() - start
    cost = get_session_cost()
    score = round(correct / total * 100, 2)
    print(f"  {method_name:30s} | {score:6.2f}% | ${cost:.4f} | {elapsed:.0f}s | {correct}/{total}")
    return score


def extract_code_completion(response: str, prompt: str) -> str:
    """Extract just the function body from the LLM response."""
    # Strategy 1: Look for code in markdown blocks
    code_match = re.search(r'```(?:python)?\s*(.*?)\s*```', response, re.DOTALL)
    if code_match:
        code = code_match.group(1)
        # If code contains the full function, extract just the body
        if 'def ' in code:
            # Find where the function body starts
            lines = code.split('\n')
            body_lines = []
            in_body = False
            for line in lines:
                if in_body:
                    body_lines.append(line)
                elif line.strip().startswith('def '):
                    in_body = True
                    # Skip lines until we find the body (after the docstring)
                elif line.strip() == '"""' or line.strip() == "'''":
                    if in_body:
                        continue
            if body_lines:
                return '\n'.join(body_lines)
        return code

    # Strategy 2: The response IS the code (no markdown)
    # Check if it looks like Python code
    lines = response.strip().split('\n')
    code_lines = [l for l in lines if not l.strip().startswith('#') or l.strip().startswith('    ')]

    # If the response starts with the function definition, strip it
    if code_lines and code_lines[0].strip().startswith('def '):
        # Skip function signature and docstring
        i = 1
        # Skip docstring
        if i < len(code_lines) and ('"""' in code_lines[i] or "'''" in code_lines[i]):
            i += 1
            while i < len(code_lines) and ('"""' not in code_lines[i] and "'''" not in code_lines[i]):
                i += 1
            i += 1
        return '\n'.join(code_lines[i:])

    return response


def main():
    samples = load_humaneval()
    print(f"=== HumanEval Improvement (gpt-4o-mini, n={len(samples)}) ===")
    print(f"Published: MaAS=92.85%, AFlow=90.9%")
    print(f"\n{'Method':30s} | {'Score':8s} | {'Cost':8s} | {'Time':5s} | Correct")
    print("-" * 70)

    # Strategy 1: Simple completion prompt
    def strategy_simple(prompt):
        result = call_llm(
            prompt=f"Complete this Python function. Return ONLY the function body:\n\n{prompt}",
            system="You are an expert Python programmer. Return ONLY code, no explanations.",
            model=MODEL, temperature=0.0, max_tokens=1024,
        )
        return result["content"]

    # Strategy 2: Direct continuation (no system prompt, treat as code completion)
    def strategy_continue(prompt):
        result = call_llm(
            prompt=prompt,
            system="Complete the Python function. Output ONLY the function body code that comes after the docstring. No markdown. No explanation.",
            model=MODEL, temperature=0.0, max_tokens=1024,
        )
        return result["content"]

    # Strategy 3: Think then code
    def strategy_think_code(prompt):
        # First think
        think = call_llm(
            prompt=f"Analyze this function signature and docstring. What approach should we take?\n\n{prompt}",
            system="Briefly plan your approach in 2-3 sentences.",
            model=MODEL, temperature=0.0, max_tokens=256,
        )
        # Then code
        result = call_llm(
            prompt=f"{prompt}\n\n# Plan: {think['content']}\n\nComplete the function body. Return ONLY code:",
            system="Write clean, correct Python. Output ONLY the function body after the docstring.",
            model=MODEL, temperature=0.0, max_tokens=1024,
        )
        return result["content"]

    # Strategy 4: Generate + test + fix
    def strategy_self_fix(prompt):
        # Generate
        result = call_llm(
            prompt=f"Complete this Python function body:\n\n{prompt}",
            system="Expert Python programmer. Return ONLY the code body.",
            model=MODEL, temperature=0.0, max_tokens=1024,
        )
        code = result["content"]

        # Quick check if it looks right
        if 'return' not in code and 'yield' not in code and 'print' not in code:
            # Retry with hint
            result = call_llm(
                prompt=f"Complete this function. Make sure to include a return statement:\n\n{prompt}",
                system="Return ONLY the function body code.",
                model=MODEL, temperature=0.0, max_tokens=1024,
            )
            code = result["content"]

        return code

    eval_humaneval_v2(strategy_simple, samples, "simple_completion")
    eval_humaneval_v2(strategy_continue, samples, "direct_continue")
    eval_humaneval_v2(strategy_think_code, samples, "think_then_code")
    eval_humaneval_v2(strategy_self_fix, samples, "generate_and_fix")


if __name__ == "__main__":
    main()
