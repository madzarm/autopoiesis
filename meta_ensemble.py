#!/usr/bin/env python3
"""Meta-Ensemble — Router over Best Agents from Each Approach.

Approach 9: Instead of finding one best agent, use ALL discovered agents.
Each approach found different optimal architectures. A meta-router classifies
the problem and routes to the best specialist:
- Math problems → DAG-Evolve's complex pipeline (best GSM8K)
- Code problems → simple generate (best HumanEval)
- Ambiguous → LLM-Architect's adaptive design

This is an "ensemble of search algorithms" — combining the outputs of
different ADAS approaches rather than different agents within one approach.

Key novelty: The search space IS the set of discovered architectures.
Rather than building new architectures, we learn how to combine existing ones.
"""

import json
import re
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm import call_llm, CHEAP, MID, get_session_cost, reset_cost_tracking
from evaluate import extract_number, extract_math_answer, normalize_math_answer, load_gsm8k, load_humaneval
from genesis import Genome, Stage, execute_genome, fast_eval


# ═══════════════════════════════════════════════════════════════
# DISCOVERED AGENTS — best from each approach
# ═══════════════════════════════════════════════════════════════

DISCOVERED_AGENTS = {
    "simple_gen": Genome(name="simple_gen", model=CHEAP, stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="Think step by step. Provide a clear answer."),
    ]),
    "code_solver": Genome(name="code_solver", model=CHEAP, stages=[
        Stage(action="generate_code", temperature=0.0, condition="always"),
    ]),
    "gen_code": Genome(name="gen_code", model=CHEAP, stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="You are a world-class mathematician. Solve with rigor. Answer after ####."),
        Stage(action="generate_code", temperature=0.0, condition="always"),
    ]),
    "adaptive": Genome(name="adaptive", model=CHEAP, stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="Solve carefully step by step. Provide a clear final answer."),
        Stage(action="generate", temperature=0.4, condition="low_confidence",
              system_prompt="Try a different approach. Think from scratch."),
    ]),
    "code_gen": Genome(name="code_gen", model=CHEAP, stages=[
        Stage(action="generate_code", temperature=0.0, condition="always"),
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="Think step by step. Provide a clear answer."),
    ]),
    "full_pipeline": Genome(name="full_pipeline", model=CHEAP, stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="Think step by step. Answer after ####."),
        Stage(action="generate_code", temperature=0.0, condition="always"),
        Stage(action="verify", condition="always"),
        Stage(action="repair", condition="after_failure"),
        Stage(action="vote", condition="always"),
    ]),
}


# ═══════════════════════════════════════════════════════════════
# PROBLEM CLASSIFIER — route to best agent
# ═══════════════════════════════════════════════════════════════

def classify_and_route(problem: str, benchmark: str) -> Genome:
    """Classify a problem and route to the best specialist.

    Simple heuristic routing based on problem features.
    """
    text_lower = problem.lower()

    if benchmark == "humaneval":
        # Code tasks → code_gen (code first, then generate for backup)
        return DISCOVERED_AGENTS["code_gen"]
    elif benchmark == "gsm8k":
        # Math tasks → adaptive design
        problem_len = len(problem)
        if problem_len < 200:
            return DISCOVERED_AGENTS["gen_code"]  # Simple problems
        else:
            return DISCOVERED_AGENTS["adaptive"]   # Complex problems
    else:
        return DISCOVERED_AGENTS["adaptive"]


def oracle_route(problem: str, sample: dict, benchmark: str,
                 agents: dict) -> tuple:
    """Oracle routing — try all agents, return the best result.

    This gives us the upper bound for what a perfect router could achieve.
    """
    best_correct = False
    best_agent = None

    for name, genome in agents.items():
        response = execute_genome(genome, problem)

        if benchmark == "gsm8k":
            predicted = extract_number(response)
            gold = sample["gold_answer"]
            correct = predicted is not None and abs(predicted - gold) < 1e-6
        elif benchmark == "humaneval":
            prompt = sample["prompt"]
            body = re.sub(r'```python\s*', '', response)
            body = re.sub(r'```\s*', '', body)
            lines = body.split('\n')
            if lines and lines[0].strip().startswith('def '):
                i = 1
                if i < len(lines) and ('"""' in lines[i] or "'''" in lines[i]):
                    i += 1
                    while i < len(lines) and '"""' not in lines[i] and "'''" not in lines[i]:
                        i += 1
                    i += 1
                body = '\n'.join(lines[i:])
            full = sample["prompt"] + body + "\n" + sample["test"] + f"\ncheck({sample['entry_point']})"
            try:
                exec(full, {})
                correct = True
            except:
                correct = False
        else:
            correct = False

        if correct:
            best_correct = True
            best_agent = name
            break  # Found a correct answer, no need to try more

    return best_correct, best_agent


# ═══════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════

def run_meta_ensemble(
    n_gsm: int = 50,
    n_he: int = 20,
    seed: int = 42,
):
    """Run meta-ensemble evaluation."""
    gsm_samples = load_gsm8k(n=n_gsm, seed=seed)
    he_samples = load_humaneval(n=n_he, seed=seed)

    print(f"═══ Meta-Ensemble ═══")
    print(f"GSM8K: {len(gsm_samples)}, HumanEval: {len(he_samples)}")
    print(f"Agents: {len(DISCOVERED_AGENTS)}")
    print()

    # Evaluate individual agents
    print("── Individual Agent Performance ──")
    for name, genome in DISCOVERED_AGENTS.items():
        reset_cost_tracking()
        gsm_result = fast_eval(genome, gsm_samples, "gsm8k")
        he_result = fast_eval(genome, he_samples, "humaneval")
        cost = get_session_cost()
        avg = (gsm_result["score"] + he_result["score"]) / 2
        print(f"  {name:20s} | GSM={gsm_result['score']:5.1f}% HE={he_result['score']:5.1f}% "
              f"avg={avg:.1f}% cost=${cost:.4f}")

    # Heuristic routing
    print(f"\n── Heuristic Routing ──")
    reset_cost_tracking()

    gsm_correct = 0
    for sample in gsm_samples:
        agent = classify_and_route(sample["question"], "gsm8k")
        response = execute_genome(agent, sample["question"])
        predicted = extract_number(response)
        gold = sample["gold_answer"]
        if predicted is not None and abs(predicted - gold) < 1e-6:
            gsm_correct += 1

    he_correct = 0
    for sample in he_samples:
        agent = classify_and_route(sample["prompt"], "humaneval")
        response = execute_genome(agent, f"Complete this Python function body:\n\n{sample['prompt']}")
        body = re.sub(r'```python\s*', '', response)
        body = re.sub(r'```\s*', '', body)
        lines = body.split('\n')
        if lines and lines[0].strip().startswith('def '):
            i = 1
            if i < len(lines) and ('"""' in lines[i] or "'''" in lines[i]):
                i += 1
                while i < len(lines) and '"""' not in lines[i] and "'''" not in lines[i]:
                    i += 1
                i += 1
            body = '\n'.join(lines[i:])
        full = sample["prompt"] + body + "\n" + sample["test"] + f"\ncheck({sample['entry_point']})"
        try:
            exec(full, {})
            he_correct += 1
        except:
            pass

    gsm_score = round(gsm_correct / len(gsm_samples) * 100, 1)
    he_score = round(he_correct / len(he_samples) * 100, 1)
    avg_score = (gsm_score + he_score) / 2

    print(f"  Heuristic routing: GSM={gsm_score}% HE={he_score}% avg={avg_score:.1f}%")

    # Oracle routing (upper bound)
    print(f"\n── Oracle Routing (upper bound) ──")

    # Subset for oracle (expensive — tries all agents)
    oracle_gsm = 0
    oracle_n = min(20, len(gsm_samples))
    for sample in gsm_samples[:oracle_n]:
        correct, agent = oracle_route(sample["question"], sample, "gsm8k", DISCOVERED_AGENTS)
        if correct:
            oracle_gsm += 1

    oracle_he = 0
    oracle_he_n = min(10, len(he_samples))
    for sample in he_samples[:oracle_he_n]:
        correct, agent = oracle_route(
            f"Complete this Python function body:\n\n{sample['prompt']}", sample,
            "humaneval", DISCOVERED_AGENTS)
        if correct:
            oracle_he += 1

    oracle_gsm_pct = round(oracle_gsm / oracle_n * 100, 1)
    oracle_he_pct = round(oracle_he / oracle_he_n * 100, 1)
    oracle_avg = (oracle_gsm_pct + oracle_he_pct) / 2
    print(f"  Oracle routing: GSM={oracle_gsm_pct}%/{oracle_n} HE={oracle_he_pct}%/{oracle_he_n} avg={oracle_avg:.1f}%")

    print(f"\n{'═'*60}")
    print(f"META-ENSEMBLE COMPLETE")
    print(f"{'═'*60}")
    print(f"Heuristic: GSM={gsm_score}% HE={he_score}% avg={avg_score:.1f}%")
    print(f"Oracle: GSM={oracle_gsm_pct}%/{oracle_n} HE={oracle_he_pct}%/{oracle_he_n} avg={oracle_avg:.1f}%")

    return {"gsm8k": gsm_score, "humaneval": he_score, "avg": avg_score,
            "oracle_avg": oracle_avg}


if __name__ == "__main__":
    run_meta_ensemble()
