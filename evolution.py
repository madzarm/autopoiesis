#!/usr/bin/env python3
"""AIDE Evolutionary Search — full evolutionary loop with error-trace-conditioned mutation.

This is the core novel contribution: instead of random mutations or blind LLM proposals,
AIDE analyzes WHY agents fail and proposes targeted fixes.

Inspired by the adaptive immune system:
- Population = immune repertoire (diverse configs)
- Evaluation = antigen exposure (test against problems)
- Error analysis = antigen recognition (identify failure modes)
- Mutation = somatic hypermutation (targeted changes based on failure analysis)
- Selection = clonal selection (keep what works, amplify successful variants)
- Archive = immune memory (remember proven solutions)
"""

import json
import random
import time
from datetime import datetime, timezone
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from agents import AgentConfig, run_agent
from evaluate import (
    load_gsm8k, load_drop, evaluate_math_accuracy, evaluate_drop_f1,
    extract_number,
)
from archive import add_to_archive, get_best, get_archive_summary, load_archive
from search import (
    random_config, mutate_config, crossover_configs,
    llm_propose_config, llm_mutate_config, SEARCH_SPACE,
)
from llm import call_llm, STRONG, MID, CHEAP, get_session_cost, reset_cost_tracking


def error_trace_mutation(
    config: AgentConfig,
    score: float,
    error_details: list[dict],
    model: str = MID,
) -> AgentConfig:
    """The core AIDE innovation: error-trace-conditioned mutation.

    Analyzes specific failures and proposes targeted config changes.
    Analogous to somatic hypermutation guided by antigen exposure.
    """
    system = (
        "You are an expert AI system designer. You analyze why an agent failed on specific "
        "problems and propose targeted configuration changes to fix those failures. "
        "You must return a valid JSON agent configuration."
    )

    config_json = json.dumps(config.model_dump(), indent=2)
    space_json = json.dumps(SEARCH_SPACE, indent=2)

    # Build error analysis
    error_analysis = []
    for err in error_details[:5]:  # Top 5 errors
        q = err.get("question", "")[:300]
        gold = err.get("gold", "N/A")
        pred = err.get("predicted", "N/A")
        error_analysis.append(
            f"Question: {q}\n"
            f"Expected: {gold}, Got: {pred}\n"
            f"Error type: {'arithmetic' if pred is not None else 'parsing/no-answer'}"
        )

    prompt = f"""## Current Agent Configuration (scored {score:.1f}%)
{config_json}

## Failure Analysis
The agent got these questions WRONG:

{"---".join(error_analysis) if error_analysis else "No specific errors available."}

## Common Failure Patterns
Analyze the errors above. What kinds of mistakes is this agent making?
- Arithmetic errors? → Consider adding self_check or self_refine reflection
- Misunderstanding the problem? → Consider decompose reasoning or clearer instructions
- Inconsistent answers? → Consider majority_vote ensemble
- Not showing final answer? → Fix output_format instructions
- Over-complicated approach? → Simplify the pipeline

## Valid Configuration Space
{space_json}

Additional fields:
- persona: string (system prompt role)
- custom_instructions: string (appended to prompt)
- model: must be "{CHEAP}"

## Your Task
1. Diagnose the failure mode from the errors
2. Propose TARGETED changes (1-3 fields) to fix the specific issue
3. Return the COMPLETE modified JSON config

Return ONLY valid JSON. Do NOT change the model field.
"""

    result = call_llm(
        prompt=prompt,
        system=system,
        model=model,
        temperature=0.6,
        max_tokens=2048,
        json_mode=True,
    )

    try:
        data = json.loads(result["content"])
        data["model"] = CHEAP
        return AgentConfig(**data)
    except Exception:
        # Fallback: simple random mutation
        return mutate_config(config, n_mutations=2)


def tournament_select(population: list[dict], k: int = 3) -> dict:
    """Tournament selection: pick k random, return the best."""
    tournament = random.sample(population, min(k, len(population)))
    return max(tournament, key=lambda x: x["score"])


def diversity_distance(c1: AgentConfig, c2: AgentConfig) -> float:
    """Compute diversity distance between two configs."""
    dist = 0
    for field in SEARCH_SPACE:
        if getattr(c1, field) != getattr(c2, field):
            dist += 1
    return dist / len(SEARCH_SPACE)


def run_evolution(
    benchmark: str = "gsm8k",
    n_samples: int = 50,
    population_size: int = 6,
    generations: int = 10,
    elite_size: int = 2,
    mutation_rate: float = 0.7,
    crossover_rate: float = 0.2,
    random_rate: float = 0.1,
    seed: int = 42,
):
    """Run the full evolutionary search loop.

    Strategy per generation:
    - Keep top `elite_size` configs unchanged
    - Generate new configs via:
      - Error-trace mutation (mutation_rate)
      - Crossover of top configs (crossover_rate)
      - Random exploration (random_rate)
    """
    # Load benchmark
    if benchmark == "gsm8k":
        samples = load_gsm8k(split="test", n=n_samples, seed=seed)
    elif benchmark == "drop":
        samples = load_drop(split="validation", n=n_samples, seed=seed)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    print(f"=== AIDE Evolution ===")
    print(f"Benchmark: {benchmark}, Samples: {len(samples)}")
    print(f"Population: {population_size}, Generations: {generations}")
    print(f"Mutation: {mutation_rate}, Crossover: {crossover_rate}, Random: {random_rate}")
    print()

    # Initialize population with diverse configs
    population = []
    init_configs = [
        AgentConfig(name="seed_cot", reasoning="cot", output_format="step_numbered",
                    model=CHEAP, temperature=0.0),
        AgentConfig(name="seed_decompose", reasoning="decompose", planning="divide_conquer",
                    output_format="step_numbered", model=CHEAP, temperature=0.0),
        AgentConfig(name="seed_cot_refine", reasoning="cot", reflection="self_refine",
                    output_format="step_numbered", model=CHEAP, temperature=0.0),
    ]
    # Fill rest with random
    while len(init_configs) < population_size:
        init_configs.append(random_config(f"seed_random_{len(init_configs)}"))

    # Evaluate initial population in parallel
    print("--- Evaluating initial population ---")
    for config in init_configs[:population_size]:
        reset_cost_tracking()
        if benchmark in ("gsm8k",):
            agent_fn = lambda q, c=config: run_agent(c, q)
            result = evaluate_math_accuracy(agent_fn, samples, benchmark_name=benchmark)
        elif benchmark == "drop":
            agent_fn = lambda p, q, c=config: run_agent(c, q, passage=p)
            result = evaluate_drop_f1(agent_fn, samples)

        entry = {
            "config": config,
            "score": result["score"],
            "cost": result.get("cost_usd", 0),
            "details": result.get("details", []),
        }
        population.append(entry)
        add_to_archive(config, benchmark, result["score"], result.get("cost_usd", 0))
        print(f"  {config.name}: {result['score']}%")

    best_ever = max(population, key=lambda x: x["score"])
    print(f"\nBest initial: {best_ever['config'].name} = {best_ever['score']}%")

    # Evolution loop
    for gen in range(generations):
        print(f"\n{'='*50}")
        print(f"Generation {gen+1}/{generations}")
        print(f"{'='*50}")

        # Sort by score
        population.sort(key=lambda x: x["score"], reverse=True)

        # Keep elites
        new_population = population[:elite_size]
        print(f"Elites: {', '.join(f'{e['config'].name}={e['score']}%' for e in new_population)}")

        # Generate new configs
        while len(new_population) < population_size:
            roll = random.random()

            if roll < mutation_rate:
                # Error-trace mutation (the novel part!)
                parent = tournament_select(population)
                errors = [d for d in parent["details"] if not d.get("correct", True)]
                error_examples = []
                for e in errors[:5]:
                    idx = e.get("idx", 0)
                    if idx < len(samples):
                        error_examples.append({
                            "question": samples[idx].get("question", "")[:300],
                            "gold": e.get("gold", ""),
                            "predicted": e.get("predicted", ""),
                        })

                child = error_trace_mutation(
                    parent["config"],
                    parent["score"],
                    error_examples,
                )
                child.name = f"mut_g{gen+1}_{len(new_population)}"
                print(f"  Mutation from {parent['config'].name}")

            elif roll < mutation_rate + crossover_rate:
                # Crossover
                p1 = tournament_select(population)
                p2 = tournament_select(population)
                child = crossover_configs(p1["config"], p2["config"])
                child.name = f"cross_g{gen+1}_{len(new_population)}"
                print(f"  Crossover: {p1['config'].name} x {p2['config'].name}")

            else:
                # Random exploration
                child = random_config(f"rand_g{gen+1}_{len(new_population)}")
                print(f"  Random exploration")

            child.model = CHEAP  # Enforce

            # Evaluate child
            reset_cost_tracking()
            try:
                if benchmark in ("gsm8k",):
                    agent_fn = lambda q, c=child: run_agent(c, q)
                    result = evaluate_math_accuracy(agent_fn, samples, benchmark_name=benchmark)
                elif benchmark == "drop":
                    agent_fn = lambda p, q, c=child: run_agent(c, q, passage=p)
                    result = evaluate_drop_f1(agent_fn, samples)

                entry = {
                    "config": child,
                    "score": result["score"],
                    "cost": result.get("cost_usd", 0),
                    "details": result.get("details", []),
                }
                new_population.append(entry)
                add_to_archive(child, benchmark, result["score"], result.get("cost_usd", 0))
                print(f"    → {child.name}: {result['score']}%")

                if result["score"] > best_ever["score"]:
                    best_ever = entry
                    print(f"    *** NEW BEST: {result['score']}% ***")

            except Exception as e:
                print(f"    → CRASHED: {e}")
                # Add a placeholder with 0 score
                new_population.append({
                    "config": child,
                    "score": 0.0,
                    "cost": 0.0,
                    "details": [],
                })

        population = new_population

        # Log generation stats
        scores = [p["score"] for p in population]
        print(f"\nGen {gen+1} stats: best={max(scores):.1f}%, avg={sum(scores)/len(scores):.1f}%, worst={min(scores):.1f}%")
        print(f"Best ever: {best_ever['config'].name} = {best_ever['score']}%")

    # Final results
    print(f"\n{'='*50}")
    print(f"EVOLUTION COMPLETE")
    print(f"{'='*50}")
    print(f"Best config: {best_ever['config'].name}")
    print(f"Best score: {best_ever['score']}%")
    print(f"\nConfig details:")
    print(json.dumps(best_ever["config"].model_dump(), indent=2))

    # Save best config
    with open("best_evolved_config.json", "w") as f:
        json.dump(best_ever["config"].model_dump(), f, indent=2)

    return best_ever


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="gsm8k")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--pop", type=int, default=6)
    parser.add_argument("--gens", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_evolution(
        benchmark=args.benchmark,
        n_samples=args.n,
        population_size=args.pop,
        generations=args.gens,
        seed=args.seed,
    )
