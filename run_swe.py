#!/usr/bin/env python3
"""Run SWE-bench experiments — baseline + interactive agent + evolutionary search.

Usage:
    # Run baseline (one-shot) on 5 instances
    python3 run_swe.py --mode baseline --n 5

    # Run interactive agent on 5 instances
    python3 run_swe.py --mode interactive --n 5

    # Run evolutionary search on mini
    python3 run_swe.py --mode evolve --split mini --evo-depth 3

    # Evaluate predictions
    python3 run_swe.py --mode eval --predictions swe_predictions.jsonl
"""

import os
import sys
import json
import time
import functools

# Force unbuffered output for background runs
print = functools.partial(print, flush=True)
import argparse
import tempfile
import shutil
from datetime import datetime

# Use system Python 3.12 packages
sys.path.insert(0, "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages")

from datasets import load_dataset
from swe_llm import get_session_stats, reset_cost_tracking, AGENT_MODEL
from swe_agent import solve_instance, DEFAULT_CONFIG
from swe_interactive_agent import solve_interactive, DEFAULT_INTERACTIVE_CONFIG
from swe_eval import format_predictions, quick_proxy_eval, save_results
from swe_evolve import EvolutionController, SEED_CONFIGS
from swe_meta_evolve import MetaEvolver


def load_swe_bench(split: str = "verified", n: int = None, offset: int = 0) -> list:
    """Load SWE-bench dataset."""
    dataset_map = {
        "verified": "SWE-bench/SWE-bench_Verified",
        "lite": "SWE-bench/SWE-bench_Lite",
        "mini": "MariusHobbhahn/swe-bench-verified-mini",
    }

    dataset_name = dataset_map.get(split, split)
    print(f"Loading {dataset_name}...")
    ds = load_dataset(dataset_name, split="test")
    print(f"Loaded {len(ds)} instances")

    instances = []
    for i in range(len(ds)):
        inst = dict(ds[i])
        instances.append(inst)

    if offset:
        instances = instances[offset:]
    if n:
        instances = instances[:n]

    print(f"Using {len(instances)} instances (offset={offset})")
    return instances


def run_baseline(instances: list, config: dict = None, work_dir: str = None) -> list:
    """Run baseline agent on instances."""
    if config is None:
        config = DEFAULT_CONFIG.copy()

    results = []
    total = len(instances)

    for i, inst in enumerate(instances):
        instance_id = inst["instance_id"]
        print(f"\n[{i+1}/{total}] {instance_id}")

        inst_dir = os.path.join(work_dir, instance_id.replace("/", "__"))
        os.makedirs(inst_dir, exist_ok=True)

        try:
            result = solve_instance(inst, config, inst_dir)
            patch = result.get("model_patch", "")
            print(f"  Patch: {len(patch)} chars, Error: {result.get('error', 'none')}")
            results.append(result)
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                "instance_id": instance_id,
                "model_name_or_path": config.get("name", "baseline"),
                "model_patch": "",
                "error": str(e),
            })

        # Print cost periodically
        if (i + 1) % 5 == 0:
            stats = get_session_stats()
            print(f"\n  Session cost: ${stats['total_cost_usd']:.4f} ({stats['call_count']} calls)")

        # Cleanup instance dir to save space
        if os.path.exists(inst_dir):
            shutil.rmtree(inst_dir, ignore_errors=True)

    return results


def run_interactive(instances: list, config: dict = None, work_dir: str = None) -> list:
    """Run interactive agent on instances."""
    if config is None:
        config = DEFAULT_INTERACTIVE_CONFIG.copy()

    results = []
    total = len(instances)

    for i, inst in enumerate(instances):
        instance_id = inst["instance_id"]
        print(f"\n[{i+1}/{total}] {instance_id}")

        inst_dir = os.path.join(work_dir, instance_id.replace("/", "__"))
        os.makedirs(inst_dir, exist_ok=True)

        try:
            result = solve_interactive(inst, config, inst_dir)
            patch = result.get("model_patch", "")
            turns = result.get("turns_used", 0)
            cost = result.get("total_cost", 0)
            print(f"  Patch: {len(patch)} chars, Turns: {turns}, Cost: ${cost:.4f}")
            results.append(result)
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                "instance_id": instance_id,
                "model_name_or_path": "interactive_agent",
                "model_patch": "",
                "error": str(e),
            })

        # Print cost periodically
        if (i + 1) % 5 == 0:
            stats = get_session_stats()
            print(f"\n  Session cost: ${stats['total_cost_usd']:.4f} ({stats['call_count']} calls)")

        # Cleanup instance dir to save space
        if os.path.exists(inst_dir):
            shutil.rmtree(inst_dir, ignore_errors=True)

    return results


def run_meta_evolve(instances: list, work_dir: str = None,
                    evo_depth: int = 3, pop_size: int = 5) -> list:
    """Run meta-evolutionary search with interactive agents."""
    evolver = MetaEvolver(
        population_size=pop_size,
        evolution_depth=evo_depth,
        meta_model=AGENT_MODEL,
    )

    results = []
    total = len(instances)

    for i, inst in enumerate(instances):
        instance_id = inst["instance_id"]
        print(f"\n[{i+1}/{total}] Meta-evolving for {instance_id}")

        inst_dir = os.path.join(work_dir, instance_id.replace("/", "__"))
        os.makedirs(inst_dir, exist_ok=True)

        try:
            result = evolver.solve_with_evolution(inst, inst_dir)
            patch = result.get("model_patch", "")
            print(f"  Best patch: {len(patch)} chars")
            results.append(result)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "instance_id": instance_id,
                "model_name_or_path": "meta_evolve",
                "model_patch": "",
                "error": str(e),
            })

        # Print progress
        if (i + 1) % 3 == 0:
            stats = get_session_stats()
            print(f"\n  Session: ${stats['total_cost_usd']:.4f} ({stats['call_count']} calls)")
            print(f"  {evolver.get_summary()}")

        # Cleanup
        if os.path.exists(inst_dir):
            shutil.rmtree(inst_dir, ignore_errors=True)

    # Final summary
    print(f"\n{evolver.get_summary()}")
    return results


def run_evolve(instances: list, work_dir: str = None,
               evo_depth: int = 3, pop_size: int = 5) -> list:
    """Run evolutionary search on instances."""
    controller = EvolutionController(
        population_size=pop_size,
        evolution_depth=evo_depth,
        meta_model=AGENT_MODEL,  # Use Sonnet as meta-model too (cheaper)
    )

    results = []
    total = len(instances)

    for i, inst in enumerate(instances):
        instance_id = inst["instance_id"]
        print(f"\n[{i+1}/{total}] Evolving for {instance_id}")

        inst_dir = os.path.join(work_dir, instance_id.replace("/", "__"))
        os.makedirs(inst_dir, exist_ok=True)

        try:
            result = controller.evolve_for_instance(inst, inst_dir)
            patch = result.get("model_patch", "")
            print(f"  Best patch: {len(patch)} chars")
            results.append(result)
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                "instance_id": instance_id,
                "model_name_or_path": "evolve",
                "model_patch": "",
                "error": str(e),
            })

        # Print progress
        if (i + 1) % 5 == 0:
            stats = get_session_stats()
            print(f"\n  Session: ${stats['total_cost_usd']:.4f} ({stats['call_count']} calls)")
            print(f"  {controller.get_population_summary()}")

        # Cleanup
        if os.path.exists(inst_dir):
            shutil.rmtree(inst_dir, ignore_errors=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="SWE-bench ADAS experiments")
    parser.add_argument("--mode", choices=["baseline", "interactive", "meta_evolve", "evolve", "eval"], required=True)
    parser.add_argument("--split", default="verified", help="verified, lite, or mini")
    parser.add_argument("--n", type=int, default=None, help="Number of instances to process")
    parser.add_argument("--offset", type=int, default=0, help="Starting offset")
    parser.add_argument("--evo-depth", type=int, default=3, help="Evolution depth per instance")
    parser.add_argument("--pop-size", type=int, default=5, help="Population size")
    parser.add_argument("--predictions", type=str, help="Path to predictions file (for eval mode)")
    parser.add_argument("--config", type=str, default=None,
                       help="Seed config name (agentless_simple, multi_candidate_3, etc)")
    parser.add_argument("--work-dir", type=str, default=None, help="Working directory")
    args = parser.parse_args()

    # Set up API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    reset_cost_tracking()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Working directory
    work_dir = args.work_dir or tempfile.mkdtemp(prefix=f"swe_{args.mode}_")
    os.makedirs(work_dir, exist_ok=True)
    print(f"Work dir: {work_dir}")

    if args.mode == "eval":
        if not args.predictions:
            print("ERROR: --predictions required for eval mode")
            sys.exit(1)
        # Just format and validate predictions
        with open(args.predictions) as f:
            if args.predictions.endswith(".jsonl"):
                results = [json.loads(line) for line in f]
            else:
                results = json.load(f)
        instances = load_swe_bench(args.split, args.n, args.offset)
        proxy = quick_proxy_eval(results, instances)
        print(f"\nProxy evaluation:")
        for k, v in proxy.items():
            print(f"  {k}: {v}")
        return

    # Load data
    instances = load_swe_bench(args.split, args.n, args.offset)

    # Select config for baseline mode
    config = None
    if args.mode == "baseline" and args.config:
        for seed in SEED_CONFIGS:
            if seed["name"] == args.config:
                config = seed.copy()
                break
        if not config:
            print(f"Unknown config: {args.config}")
            print(f"Available: {[s['name'] for s in SEED_CONFIGS]}")
            sys.exit(1)

    # Run
    start = time.time()

    if args.mode == "baseline":
        results = run_baseline(instances, config, work_dir)
    elif args.mode == "interactive":
        results = run_interactive(instances, config=None, work_dir=work_dir)
    elif args.mode == "meta_evolve":
        results = run_meta_evolve(instances, work_dir, args.evo_depth, args.pop_size)
    elif args.mode == "evolve":
        results = run_evolve(instances, work_dir, args.evo_depth, args.pop_size)

    elapsed = time.time() - start

    # Save results
    results_path = f"swe_results_{args.mode}_{timestamp}.json"
    save_results(results, results_path)

    # Format predictions
    pred_path = f"swe_predictions_{args.mode}_{timestamp}.jsonl"
    format_predictions(results, pred_path)

    # Proxy evaluation
    proxy = quick_proxy_eval(results, instances)

    # Summary
    stats = get_session_stats()
    print(f"\n{'='*60}")
    print(f"SUMMARY — {args.mode} on {args.split} ({len(instances)} instances)")
    print(f"{'='*60}")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Cost: ${stats['total_cost_usd']:.4f}")
    print(f"API calls: {stats['call_count']}")
    print(f"Tokens: {stats['total_input_tokens']:,} in / {stats['total_output_tokens']:,} out")
    print(f"\nProxy eval:")
    for k, v in proxy.items():
        print(f"  {k}: {v}")
    print(f"\nResults: {results_path}")
    print(f"Predictions: {pred_path}")


if __name__ == "__main__":
    main()
