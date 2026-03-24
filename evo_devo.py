#!/usr/bin/env python3
"""Evo-Devo — Evolving Developmental Programs that Generate Agent Architectures.

Approach 10: Instead of searching over agent configs (approaches 1-9), search over
PROGRAMS that generate configs. The genotype is a small Python program; the phenotype
is the agent architecture it produces.

Key novelty:
- Genotype-phenotype separation: the evolved entity is a PROGRAM, not a config
- Same program can produce different architectures for different problems
- Programs can express patterns (loops, conditionals) that flat configs can't
- Programs can inspect the problem and adapt the architecture dynamically

This is the evo-devo principle: evolution searches over developmental programs,
not over organisms directly. A genome doesn't encode "have 5 fingers" — it encodes
a program that, when executed, produces fingers.

Implementation: use the LLM to generate and mutate small Python functions that
take a problem description and return a Genome.
"""

import json
import re
import random
import copy
import time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm import call_llm, STRONG, MID, CHEAP, get_session_cost, reset_cost_tracking
from genesis import Genome, Stage, execute_genome, fast_eval
from evaluate import load_gsm8k, load_humaneval


# ═══════════════════════════════════════════════════════════════
# DEVELOPMENTAL PROGRAMS
# ═══════════════════════════════════════════════════════════════

# A developmental program is a Python function that takes a problem string
# and returns a Genome. We evolve these programs.

STAGE_CLASS = """
class Stage:
    def __init__(self, action, temperature=0.0, system_prompt="", condition="always",
                 condition_threshold=0.7, terminate_if_confident=False, confidence_threshold=0.9):
        self.action = action
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.condition = condition
        self.condition_threshold = condition_threshold
        self.terminate_if_confident = terminate_if_confident
        self.confidence_threshold = confidence_threshold

class Genome:
    def __init__(self, name="", model="gpt-5.4-nano-2026-03-17", stages=None, max_candidates=5):
        self.name = name
        self.model = model
        self.stages = stages or []
        self.max_candidates = max_candidates
"""


def exec_dev_program(code: str, problem: str) -> Genome:
    """Execute a developmental program on a problem to produce a Genome."""
    try:
        local_ns = {}
        exec(STAGE_CLASS, local_ns)
        exec(code, local_ns)

        if "make_agent" not in local_ns:
            return None

        result = local_ns["make_agent"](problem)

        # Convert to real Genome
        if hasattr(result, 'stages') and result.stages:
            stages = []
            for s in result.stages:
                stages.append(Stage(
                    action=s.action,
                    temperature=getattr(s, 'temperature', 0.0),
                    system_prompt=getattr(s, 'system_prompt', ''),
                    condition=getattr(s, 'condition', 'always'),
                    condition_threshold=getattr(s, 'condition_threshold', 0.7),
                    terminate_if_confident=getattr(s, 'terminate_if_confident', False),
                    confidence_threshold=getattr(s, 'confidence_threshold', 0.9),
                ))
            return Genome(name="devo", model=CHEAP, stages=stages)
    except Exception as e:
        pass
    return None


# Seed programs
SEED_PROGRAMS = [
    # 1. Simple — always generate
    '''
def make_agent(problem):
    return Genome(stages=[
        Stage(action="generate", temperature=0.0, system_prompt="Think step by step. Provide a clear answer."),
    ])
''',
    # 2. Adaptive — code for short problems, generate for long
    '''
def make_agent(problem):
    if len(problem) < 200:
        return Genome(stages=[
            Stage(action="generate_code", temperature=0.0),
        ])
    else:
        return Genome(stages=[
            Stage(action="generate", temperature=0.0, system_prompt="Think step by step. Provide a clear answer."),
            Stage(action="generate_code", temperature=0.3, condition="low_confidence"),
        ])
''',
    # 3. Complexity-adaptive — more stages for harder problems
    '''
def make_agent(problem):
    words = len(problem.split())
    stages = [Stage(action="generate", temperature=0.0, system_prompt="Think step by step. Provide a clear answer.")]
    if words > 50:
        stages.append(Stage(action="generate_code", temperature=0.0))
    if words > 100:
        stages.append(Stage(action="verify", condition="always"))
        stages.append(Stage(action="repair", condition="after_failure"))
    return Genome(stages=stages)
''',
    # 4. Task-type detector
    '''
def make_agent(problem):
    text = problem.lower()
    if any(w in text for w in ["code", "function", "python", "program", "def "]):
        return Genome(stages=[
            Stage(action="generate", temperature=0.0, system_prompt="Write clean Python code."),
        ])
    elif any(w in text for w in ["how many", "how much", "calculate", "compute"]):
        return Genome(stages=[
            Stage(action="generate", temperature=0.0, system_prompt="Solve step by step. Answer after ####."),
            Stage(action="generate_code", temperature=0.0, condition="low_confidence"),
        ])
    else:
        return Genome(stages=[
            Stage(action="generate", temperature=0.0, system_prompt="Think step by step. Provide a clear answer."),
        ])
''',
]


# ═══════════════════════════════════════════════════════════════
# LLM-GUIDED PROGRAM EVOLUTION
# ═══════════════════════════════════════════════════════════════

def mutate_program(code: str, score: float, errors: list, model: str = STRONG) -> str:
    """Use LLM to mutate a developmental program."""
    error_str = "\n".join(f"- {e}" for e in errors[:3])

    prompt = f"""Evolve this Python function that generates agent pipelines.

Current program (scored {score:.1f}%):
```python
{code}
```

Errors on test problems:
{error_str if error_str else "None"}

The function `make_agent(problem)` should return a Genome with stages.
Available actions: "generate", "generate_code", "verify", "repair", "vote"
Available conditions: "always", "low_confidence", "after_failure", "disagreement"

The program can inspect the problem text to decide what stages to use.
Make the program BETTER at handling diverse problem types.

Return ONLY the Python code for the function (no explanation).
```python
def make_agent(problem):
    ...
```
"""
    result = call_llm(prompt=prompt, system="Python programmer evolving agent construction code.",
                      model=model, temperature=0.7, max_tokens=1024)

    # Extract code
    text = result["content"]
    match = re.search(r'```python\s*(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try to find the function directly
    match = re.search(r'(def make_agent.*)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return code  # Fallback to original


# ═══════════════════════════════════════════════════════════════
# MULTI-BENCHMARK EVALUATION FOR DEV PROGRAMS
# ═══════════════════════════════════════════════════════════════

def eval_dev_program(code: str, gsm_samples: list, he_samples: list) -> dict:
    """Evaluate a developmental program on multiple benchmarks."""
    gsm_correct = 0
    he_correct = 0
    errors = []

    # GSM8K evaluation
    for sample in gsm_samples:
        genome = exec_dev_program(code, sample["question"])
        if genome is None:
            continue
        result = fast_eval(genome, [sample], "gsm8k")
        if result["score"] >= 99:
            gsm_correct += 1
        elif result.get("errors"):
            errors.append(f"GSM: gold={sample['gold_answer']}")

    # HumanEval evaluation
    for sample in he_samples:
        genome = exec_dev_program(code, sample["prompt"])
        if genome is None:
            continue
        result = fast_eval(genome, [sample], "humaneval")
        if result["score"] >= 99:
            he_correct += 1
        elif result.get("errors"):
            errors.append(f"HE: {sample['entry_point']}")

    gsm_score = round(gsm_correct / max(len(gsm_samples), 1) * 100, 1)
    he_score = round(he_correct / max(len(he_samples), 1) * 100, 1)
    avg = (gsm_score + he_score) / 2

    return {"gsm8k": gsm_score, "humaneval": he_score, "avg": avg, "errors": errors[:5]}


# ═══════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════

def run_evo_devo(
    n_gsm: int = 20,
    n_he: int = 10,
    n_iterations: int = 15,
    seed: int = 42,
):
    """Run evo-devo search over developmental programs."""
    gsm_samples = load_gsm8k(n=n_gsm, seed=seed)
    he_samples = load_humaneval(n=n_he, seed=seed)

    print(f"═══ Evo-Devo ═══")
    print(f"GSM8K: {len(gsm_samples)}, HumanEval: {len(he_samples)}")
    print(f"Iterations: {n_iterations}")
    print()

    best_score = 0.0
    best_program = ""
    population = list(SEED_PROGRAMS)

    # Evaluate seeds
    print("── Seed Programs ──")
    scored_pop = []
    for i, code in enumerate(population):
        reset_cost_tracking()
        result = eval_dev_program(code, gsm_samples, he_samples)
        scored_pop.append((code, result))
        print(f"  seed_{i}: GSM={result['gsm8k']:5.1f}% HE={result['humaneval']:5.1f}% avg={result['avg']:.1f}%")
        if result["avg"] > best_score:
            best_score = result["avg"]
            best_program = code

    print(f"Best seed: {best_score:.1f}%\n")

    # Evolution loop
    print("── Evolution ──")
    for i in range(n_iterations):
        # Select parent
        scored_pop.sort(key=lambda x: x[1]["avg"], reverse=True)
        parent_code, parent_result = scored_pop[0]

        # Mutate
        reset_cost_tracking()
        child_code = mutate_program(parent_code, parent_result["avg"], parent_result.get("errors", []))

        # Evaluate
        result = eval_dev_program(child_code, gsm_samples, he_samples)

        marker = ""
        if result["avg"] > best_score:
            best_score = result["avg"]
            best_program = child_code
            marker = " *** NEW BEST ***"

        scored_pop.append((child_code, result))
        # Keep top 5
        scored_pop.sort(key=lambda x: x[1]["avg"], reverse=True)
        scored_pop = scored_pop[:5]

        if (i + 1) % 3 == 0 or marker:
            print(f"  [{i+1:3d}/{n_iterations}] GSM={result['gsm8k']:5.1f}% HE={result['humaneval']:5.1f}% "
                  f"avg={result['avg']:.1f}%{marker}")

    # Final
    print(f"\n{'═'*50}")
    print(f"EVO-DEVO COMPLETE")
    print(f"{'═'*50}")
    print(f"Best avg: {best_score:.1f}%")
    print(f"Best program:")
    print(best_program)

    return {"avg": best_score, "program": best_program}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--gsm", type=int, default=20)
    p.add_argument("--he", type=int, default=10)
    p.add_argument("--iters", type=int, default=15)
    args = p.parse_args()
    run_evo_devo(args.gsm, args.he, args.iters)
