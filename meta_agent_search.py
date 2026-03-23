#!/usr/bin/env python3
"""ADAS-style Meta Agent Search — LLM generates agent CODE, evaluates it, archives best.

This implements the core idea from Hu et al. (2024): instead of searching over
structured configs, let the LLM generate arbitrary Python code that defines an agent.
The agent code takes a question and returns an answer string.

Key differences from the original ADAS paper:
- We use gpt-5.4-nano as backbone (they used gpt-3.5/4)
- We add safety constraints on generated code
- We search across multiple benchmarks simultaneously
"""

import json
import re
import time
import traceback
from datetime import datetime, timezone

from llm import call_llm, STRONG, MID, CHEAP, get_session_cost, reset_cost_tracking
from evaluate import (
    load_gsm8k, load_arc, load_drop,
    evaluate_math_accuracy, evaluate_arc_accuracy, evaluate_drop_f1,
    extract_number, extract_answer_letter, extract_text_answer,
)


AGENT_CODE_TEMPLATE = '''
def forward(question: str, call_llm_fn) -> str:
    """Agent that processes a question and returns an answer string.

    Args:
        question: The question/problem to solve
        call_llm_fn: Function(prompt, system="", temperature=0.0) -> str
            Calls the LLM and returns the response text.

    Returns:
        A string containing the answer. For math, include "#### <number>".
        For multiple choice, include "#### <letter>".
        For text answers, include "#### <text>".
    """
    # YOUR CODE HERE
'''


def generate_agent_code(
    archive: list[dict],
    benchmark_results: dict,
    iteration: int,
    model: str = STRONG,
) -> str:
    """Use the meta-agent to generate new agent code."""

    # Build archive summary
    archive_str = ""
    for entry in archive[-5:]:  # Last 5 entries
        archive_str += f"\n--- Agent: {entry['name']} (avg score: {entry.get('avg_score', 0):.1f}%) ---\n"
        archive_str += f"Code:\n```python\n{entry['code'][:500]}\n```\n"
        if entry.get('scores'):
            archive_str += f"Scores: {entry['scores']}\n"
        if entry.get('errors'):
            archive_str += f"Errors: {entry['errors'][:200]}\n"

    system = (
        "You are a brilliant AI researcher designing novel agent architectures. "
        "You write Python code that defines how an agent processes questions. "
        "Your agents should be creative, effective, and use the LLM efficiently."
    )

    prompt = f"""## Task
Design a NEW agent that outperforms all previous agents. The agent is a Python function
`forward(question, call_llm_fn)` that takes a question string and an LLM calling function,
and returns an answer string.

## Available LLM Function
```python
call_llm_fn(prompt: str, system: str = "", temperature: float = 0.0) -> str
```
This calls {CHEAP} and returns the response text. Use it as many times as needed.

## Previous Agents (from archive)
{archive_str if archive_str else "No previous agents yet. Start with something creative!"}

## Performance So Far
{json.dumps(benchmark_results, indent=2) if benchmark_results else "No results yet."}

## Rules
1. The function MUST be named `forward` with exactly these args: (question, call_llm_fn)
2. Return a string. For math: include "#### <number>". For MC: include "#### <letter>".
3. You can call call_llm_fn multiple times (planning, solving, verifying, etc.)
4. Be creative! Try novel approaches: code generation, decomposition, multi-perspective debate,
   analogical reasoning, self-consistency, progressive refinement, etc.
5. Keep it under 80 lines of code.
6. Do NOT import anything except re, json, collections (already available).
7. This is iteration {iteration}. {"Try something RADICALLY different from the archive." if iteration > 3 else "Start with a strong baseline approach."}

## Output
Return ONLY the Python code for the `forward` function, wrapped in ```python ... ```.
"""

    result = call_llm(prompt=prompt, system=system, model=model,
                      temperature=0.8, max_tokens=4096)

    # Extract code
    code_match = re.search(r'```python\s*(.*?)\s*```', result["content"], re.DOTALL)
    if code_match:
        return code_match.group(1)
    return result["content"]


def compile_agent(code: str):
    """Compile agent code and return the forward function."""
    import collections

    # Create a namespace with allowed imports
    namespace = {
        're': re,
        'json': json,
        'collections': collections,
        'Counter': collections.Counter,
    }

    exec(code, namespace)

    if 'forward' not in namespace:
        raise ValueError("Code does not define a 'forward' function")

    return namespace['forward']


def evaluate_agent_code(
    forward_fn,
    benchmarks: dict[str, list],
    call_llm_fn,
) -> dict:
    """Evaluate an agent's forward function across benchmarks."""
    results = {}
    errors = []

    for bench_name, samples in benchmarks.items():
        reset_cost_tracking()
        try:
            if bench_name == "gsm8k":
                agent_fn = lambda q: forward_fn(q, call_llm_fn)
                result = evaluate_math_accuracy(agent_fn, samples, "gsm8k")
            elif bench_name == "arc":
                agent_fn = lambda q, c: forward_fn(
                    f"{q}\n\nChoices:\n{c}\n\nSelect the correct answer letter.",
                    call_llm_fn,
                )
                result = evaluate_arc_accuracy(agent_fn, samples)
            elif bench_name == "drop":
                agent_fn = lambda p, q: forward_fn(
                    f"Passage: {p}\n\nQuestion: {q}",
                    call_llm_fn,
                )
                result = evaluate_drop_f1(agent_fn, samples)
            else:
                continue

            results[bench_name] = result["score"]
            # Collect error examples
            for d in result.get("details", []):
                if not d.get("correct", True) and d.get("f1", 1.0) < 0.5:
                    errors.append(f"{bench_name}: {str(d)[:150]}")
        except Exception as e:
            results[bench_name] = 0.0
            errors.append(f"{bench_name}: CRASH - {str(e)[:100]}")

    scores = list(results.values())
    avg = sum(scores) / len(scores) if scores else 0.0

    return {
        "scores": results,
        "avg_score": round(avg, 2),
        "cost": round(get_session_cost(), 4),
        "errors": errors[:5],
    }


def make_call_llm_fn():
    """Create a simplified LLM calling function for agents."""
    def call_fn(prompt: str, system: str = "", temperature: float = 0.0) -> str:
        result = call_llm(prompt=prompt, system=system, model=CHEAP,
                         temperature=temperature, max_tokens=2048)
        return result["content"]
    return call_fn


def run_meta_agent_search(
    n_samples: int = 30,
    iterations: int = 10,
    seed: int = 42,
):
    """Run ADAS-style Meta Agent Search."""
    print("=== Meta Agent Search (ADAS-style) ===")
    print(f"Samples: {n_samples}, Iterations: {iterations}")

    # Load benchmarks
    benchmarks = {
        "gsm8k": load_gsm8k(n=n_samples, seed=seed),
        "arc": load_arc(n=n_samples, seed=seed),
    }
    print(f"Loaded: {', '.join(f'{k}={len(v)}' for k, v in benchmarks.items())}")

    call_llm_fn = make_call_llm_fn()
    archive = []
    best_score = 0.0

    for i in range(iterations):
        print(f"\n{'='*50}")
        print(f"Iteration {i+1}/{iterations}")
        print(f"{'='*50}")

        # Generate agent code
        benchmark_results = {
            "best_score": best_score,
            "archive_size": len(archive),
        }
        if archive:
            benchmark_results["best_agent"] = archive[-1]["name"] if archive else "none"
            benchmark_results["best_scores"] = archive[-1].get("scores", {})

        try:
            code = generate_agent_code(archive, benchmark_results, i, model=STRONG)
            print(f"Generated {len(code)} chars of code")

            # Compile
            forward_fn = compile_agent(code)
            print("Compiled successfully")

            # Evaluate
            eval_result = evaluate_agent_code(forward_fn, benchmarks, call_llm_fn)

            name = f"meta_agent_{i+1}"
            entry = {
                "name": name,
                "code": code,
                "avg_score": eval_result["avg_score"],
                "scores": eval_result["scores"],
                "errors": eval_result.get("errors", []),
                "cost": eval_result["cost"],
            }
            archive.append(entry)

            print(f"Scores: {eval_result['scores']}")
            print(f"Average: {eval_result['avg_score']}%")
            print(f"Cost: ${eval_result['cost']:.4f}")

            if eval_result["avg_score"] > best_score:
                best_score = eval_result["avg_score"]
                print(f"*** NEW BEST: {best_score}% ***")
                # Save best code
                with open("best_meta_agent.py", "w") as f:
                    f.write(code)

        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
            archive.append({
                "name": f"meta_agent_{i+1}_FAILED",
                "code": code if 'code' in dir() else "",
                "avg_score": 0.0,
                "scores": {},
                "errors": [str(e)],
            })

    # Final summary
    print(f"\n{'='*50}")
    print("META AGENT SEARCH COMPLETE")
    print(f"{'='*50}")
    print(f"Best score: {best_score}%")
    print(f"Archive size: {len(archive)}")

    # Print top 3
    sorted_archive = sorted(archive, key=lambda x: x["avg_score"], reverse=True)
    print("\nTop 3 agents:")
    for entry in sorted_archive[:3]:
        print(f"  {entry['name']}: {entry['avg_score']}% — {entry['scores']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_meta_agent_search(n_samples=args.n, iterations=args.iters, seed=args.seed)
