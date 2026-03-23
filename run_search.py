#!/usr/bin/env python3
"""AIDE Search Runner — runs the full search loop.

Usage:
    python run_search.py --benchmark gsm8k --n 50 --steps 5
"""

import argparse
import json
import sys
import time
import random
from datetime import datetime, timezone

from agents import AgentConfig, BASELINE_CONFIGS, run_agent
from evaluate import load_gsm8k, load_mgsm, load_drop, evaluate_math_accuracy, evaluate_drop_f1
from archive import add_to_archive, get_best, get_archive_summary, load_archive
from search import (
    random_config, mutate_config, crossover_configs,
    llm_propose_config, llm_mutate_config, get_error_examples,
    SEARCH_SPACE,
)
from llm import get_session_cost, reset_cost_tracking, CHEAP


def load_benchmark(name: str, n: int, seed: int = 42):
    if name == "gsm8k":
        return load_gsm8k(split="test", n=n, seed=seed)
    elif name == "mgsm":
        return load_mgsm(lang="en", n=n, seed=seed)
    elif name == "drop":
        return load_drop(split="validation", n=n, seed=seed)
    else:
        raise ValueError(f"Unknown benchmark: {name}")


def evaluate_config(config: AgentConfig, benchmark: str, samples: list) -> dict:
    """Evaluate a config and return results."""
    reset_cost_tracking()

    if benchmark in ("gsm8k", "mgsm"):
        agent_fn = lambda q: run_agent(config, q)
        result = evaluate_math_accuracy(agent_fn, samples, benchmark_name=benchmark)
    elif benchmark == "drop":
        agent_fn = lambda p, q: run_agent(config, q, passage=p)
        result = evaluate_drop_f1(agent_fn, samples)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    return result


def log_result(commit: str, name: str, benchmark: str, score: float,
               cost: float, status: str, desc: str, notes: str = ""):
    """Append a result to results.tsv."""
    ts = datetime.now(timezone.utc).isoformat()
    line = f"{commit}\t{ts}\t{name}\t{benchmark}\t{score}\t{cost:.4f}\t{status}\t{desc}\t{notes}\n"
    with open("results.tsv", "a") as f:
        if f.tell() == 0:
            f.write("commit\ttimestamp\texperiment_name\tbenchmark\tscore\tcost_usd\tstatus\tdescription\tnotes\n")
        f.write(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="gsm8k")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = load_benchmark(args.benchmark, args.n, args.seed)
    print(f"Loaded {len(samples)} samples for {args.benchmark}")

    archive = load_archive()
    best_score = 0.0
    if archive:
        best_entries = get_best(args.benchmark, top_k=1)
        if best_entries:
            best_score = best_entries[0]["score"]
    print(f"Current best score: {best_score}")

    stuck_count = 0

    for step in range(args.steps):
        print(f"\n{'='*60}")
        print(f"STEP {step + 1}/{args.steps}")
        print(f"{'='*60}")

        try:
            # Decide strategy
            archive_summary = get_archive_summary()

            if step < 2 or stuck_count > 2:
                # Exploration phase: LLM proposes from scratch
                print("Strategy: LLM proposal (exploration)")
                config = llm_propose_config(
                    archive_summary=archive_summary,
                    benchmark=args.benchmark,
                )
            else:
                # Exploitation phase: mutate best config
                print("Strategy: LLM-guided mutation (exploitation)")
                best_entries = get_best(args.benchmark, top_k=3)
                if best_entries:
                    # Pick a parent (weighted by score)
                    parent_data = random.choice(best_entries)
                    parent = AgentConfig(**parent_data["config"])
                    print(f"Parent: {parent.name} (score={parent_data['score']})")

                    # Get error examples from last eval
                    error_examples = []  # Will be populated from eval details

                    config = llm_mutate_config(
                        config=parent,
                        score=parent_data["score"],
                        error_examples=error_examples,
                    )
                else:
                    config = llm_propose_config(
                        archive_summary=archive_summary,
                        benchmark=args.benchmark,
                    )

            config.model = CHEAP  # Enforce cheap model

            print(f"\nEvaluating: {config.name}")
            print(f"  reasoning={config.reasoning}, planning={config.planning}")
            print(f"  reflection={config.reflection}, ensemble={config.ensemble}(n={config.ensemble_n})")
            print(f"  temp={config.temperature}, format={config.output_format}")
            if config.persona:
                print(f"  persona={config.persona[:80]}...")
            if config.custom_instructions:
                print(f"  instructions={config.custom_instructions[:80]}...")

            # Evaluate
            result = evaluate_config(config, args.benchmark, samples)

            score = result["score"]
            cost = result.get("cost_usd", 0.0)

            print(f"\nscore: {score}")
            print(f"cost_usd: {cost:.4f}")
            print(f"correct: {result.get('correct', 'N/A')}")
            print(f"total: {result['total']}")

            # Track improvement
            if score > best_score:
                print(f"*** NEW BEST: {score} (was {best_score}) ***")
                best_score = score
                stuck_count = 0
                status = "success"
            elif score == best_score:
                print(f"Equal to best ({best_score})")
                stuck_count += 1
                status = "success"
            else:
                print(f"Below best ({best_score})")
                stuck_count += 1
                status = "success"

            # Log
            log_result(
                commit="search",
                name=config.name,
                benchmark=args.benchmark,
                score=score,
                cost=cost,
                status=status,
                desc=f"r={config.reasoning} p={config.planning} ref={config.reflection} e={config.ensemble}",
                notes=f"persona={'yes' if config.persona else 'no'} stuck={stuck_count}",
            )

            # Add to archive
            add_to_archive(config, args.benchmark, score, cost)

        except Exception as e:
            print(f"error: {e}")
            import traceback
            traceback.print_exc()
            log_result(
                commit="search",
                name="error",
                benchmark=args.benchmark,
                score=0.0,
                cost=0.0,
                status="crashed",
                desc=str(e)[:200],
            )

    # Final summary
    print(f"\n{'='*60}")
    print("SEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"Best score: {best_score}")
    print(get_archive_summary())


if __name__ == "__main__":
    main()
