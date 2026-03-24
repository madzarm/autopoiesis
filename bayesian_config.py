#!/usr/bin/env python3
"""Bayesian-Config — Gaussian Process Surrogate over Agent Configuration Space.

Approach 5: Instead of population-based (Genesis), graph-based (DAG-Evolve),
tree search (MCTS-Morph), or diversity-based (Immune-QD) search:
use Bayesian Optimization with a GP surrogate model.

Key novelty:
- Encode agent configs as continuous feature vectors
- Fit a Gaussian Process to predict score from features
- Use Expected Improvement (EI) acquisition to pick next config
- Most DATA-EFFICIENT approach: each eval informs a global model
- No population, no tree, no archive — pure function optimization

This is the AutoML approach: treat agent design as black-box optimization.
Contrast with evolutionary approaches that need large populations.
"""

import json
import re
import random
import copy
import time
import math
import numpy as np
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm import call_llm, STRONG, MID, CHEAP, get_session_cost, reset_cost_tracking
from evaluate import extract_number, extract_math_answer, normalize_math_answer
from genesis import (
    Genome, Stage, execute_genome, fast_eval,
    prim_generate, prim_generate_code, prim_verify, prim_repair, prim_vote,
)


# ═══════════════════════════════════════════════════════════════
# CONFIG SPACE — discrete choices encoded as continuous features
# ═══════════════════════════════════════════════════════════════

# Each agent config is encoded as a fixed-length feature vector:
# [n_stages, has_generate, has_code, has_verify, has_repair, has_vote,
#  avg_temp, max_temp, n_conditional, frac_always, frac_generate]

FEATURE_DIM = 11

def genome_to_features(genome: Genome) -> np.ndarray:
    """Encode a genome as a continuous feature vector."""
    stages = genome.stages
    n = len(stages)
    if n == 0:
        return np.zeros(FEATURE_DIM)

    actions = [s.action for s in stages]
    temps = [s.temperature for s in stages]
    conditions = [s.condition for s in stages]

    features = np.array([
        n / 7.0,  # normalized stage count
        1.0 if "generate" in actions else 0.0,
        1.0 if "generate_code" in actions else 0.0,
        1.0 if "verify" in actions else 0.0,
        1.0 if "repair" in actions else 0.0,
        1.0 if "vote" in actions else 0.0,
        np.mean(temps),
        max(temps),
        sum(1 for c in conditions if c != "always") / max(n, 1),
        sum(1 for c in conditions if c == "always") / max(n, 1),
        sum(1 for a in actions if a in ("generate", "generate_code")) / max(n, 1),
    ])
    return features


# ═══════════════════════════════════════════════════════════════
# SIMPLE GP — minimal Gaussian Process implementation
# ═══════════════════════════════════════════════════════════════

class SimpleGP:
    """Minimal Gaussian Process regressor with RBF kernel."""

    def __init__(self, length_scale: float = 1.0, noise: float = 0.1):
        self.length_scale = length_scale
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.K_inv = None

    def _rbf_kernel(self, X1, X2):
        """RBF (Gaussian) kernel."""
        sqdist = np.sum(X1**2, axis=1, keepdims=True) + \
                 np.sum(X2**2, axis=1, keepdims=True).T - \
                 2 * X1 @ X2.T
        return np.exp(-0.5 * sqdist / self.length_scale**2)

    def fit(self, X, y):
        """Fit the GP to observed data. Normalizes y to zero mean, unit variance."""
        self.X_train = np.array(X)
        y_raw = np.array(y)
        # Normalize y for better GP behavior
        self.y_mean = np.mean(y_raw)
        self.y_std = max(np.std(y_raw), 1e-6)
        self.y_train = (y_raw - self.y_mean) / self.y_std
        K = self._rbf_kernel(self.X_train, self.X_train)
        K += self.noise**2 * np.eye(len(K))
        try:
            self.K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            self.K_inv = np.linalg.pinv(K)

    def predict(self, X):
        """Predict mean and variance at new points. Returns in original scale."""
        X = np.array(X)
        if self.X_train is None or len(self.X_train) == 0:
            return np.zeros(len(X)), np.ones(len(X))

        K_star = self._rbf_kernel(X, self.X_train)
        K_star_star = self._rbf_kernel(X, X)

        mu_norm = K_star @ self.K_inv @ self.y_train
        var_norm = np.diag(K_star_star - K_star @ self.K_inv @ K_star.T)
        var_norm = np.maximum(var_norm, 1e-6)  # Ensure positive
        # Transform back to original scale
        mu = mu_norm * self.y_std + self.y_mean
        var = var_norm * self.y_std**2
        return mu, var


def expected_improvement(mu, var, best_y, xi=0.01):
    """Expected Improvement acquisition function."""
    sigma = np.sqrt(var)
    with np.errstate(divide='ignore'):
        Z = (mu - best_y - xi) / sigma
    # Approximate normal CDF and PDF
    ei = sigma * (Z * _norm_cdf(Z) + _norm_pdf(Z))
    ei[sigma < 1e-8] = 0.0
    return ei


def _norm_cdf(x):
    """Approximate standard normal CDF."""
    return 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))


def _norm_pdf(x):
    """Standard normal PDF."""
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


# ═══════════════════════════════════════════════════════════════
# CANDIDATE GENERATION — diverse configs from different strategies
# ═══════════════════════════════════════════════════════════════

STAGE_TEMPLATES = [
    Stage(action="generate", temperature=0.0, condition="always",
          system_prompt="Think step by step. Answer after ####."),
    Stage(action="generate", temperature=0.3, condition="always",
          system_prompt="Solve carefully. Answer after ####."),
    Stage(action="generate", temperature=0.5, condition="always",
          system_prompt="Be creative. Answer after ####."),
    Stage(action="generate", temperature=0.0, condition="always",
          system_prompt="You are a world-class mathematician. Answer after ####."),
    Stage(action="generate_code", temperature=0.0, condition="always"),
    Stage(action="generate_code", temperature=0.3, condition="always"),
    Stage(action="verify", condition="always"),
    Stage(action="verify", condition="low_confidence", condition_threshold=0.7),
    Stage(action="repair", condition="after_failure", temperature=0.1),
    Stage(action="repair", condition="always", temperature=0.0),
    Stage(action="vote", condition="always"),
]


def random_genome(name: str = "rand") -> Genome:
    """Generate a random genome."""
    n_stages = random.randint(1, 5)
    stages = [copy.deepcopy(random.choice(STAGE_TEMPLATES)) for _ in range(n_stages)]
    # Ensure at least one generate
    if not any(s.action in ("generate", "generate_code") for s in stages):
        stages.insert(0, copy.deepcopy(STAGE_TEMPLATES[0]))
    return Genome(name=name, model=CHEAP, stages=stages)


def perturb_genome(genome: Genome, magnitude: float = 0.5) -> Genome:
    """Perturb a genome by making small changes."""
    new = Genome(name=genome.name + "_p", model=genome.model,
                 stages=[copy.deepcopy(s) for s in genome.stages])

    # Number of mutations proportional to magnitude
    n_muts = max(1, int(magnitude * 3))
    for _ in range(n_muts):
        roll = random.random()
        if roll < 0.3 and len(new.stages) < 6:
            # Add a stage
            pos = random.randint(0, len(new.stages))
            new.stages.insert(pos, copy.deepcopy(random.choice(STAGE_TEMPLATES)))
        elif roll < 0.5 and len(new.stages) > 1:
            # Remove a stage
            idx = random.randint(0, len(new.stages) - 1)
            new.stages.pop(idx)
        elif roll < 0.7:
            # Replace a stage
            idx = random.randint(0, len(new.stages) - 1)
            new.stages[idx] = copy.deepcopy(random.choice(STAGE_TEMPLATES))
        else:
            # Modify temperature
            idx = random.randint(0, len(new.stages) - 1)
            new.stages[idx].temperature = random.choice([0.0, 0.1, 0.3, 0.5])

    # Ensure at least one generate
    if not any(s.action in ("generate", "generate_code") for s in new.stages):
        new.stages.insert(0, copy.deepcopy(STAGE_TEMPLATES[0]))

    return new


def generate_candidates(gp: SimpleGP, observed_X: list, observed_y: list,
                        n_candidates: int = 50) -> list:
    """Generate candidate genomes using GP-guided search.

    Strategy:
    1. Generate random candidates
    2. Perturb the best observed genomes
    3. Score all by Expected Improvement
    4. Return the top ones
    """
    candidates = []

    # Random exploration
    for i in range(n_candidates // 2):
        candidates.append(random_genome(f"rand_{i}"))

    # Perturbation of best observed
    if observed_X and observed_y:
        best_idx = np.argmax(observed_y)
        best_genome_features = observed_X[best_idx]
        # We can't directly reconstruct from features, so perturb the best known genomes
        # Store genomes alongside features
        pass

    return candidates


# ═══════════════════════════════════════════════════════════════
# BAYESIAN OPTIMIZATION LOOP
# ═══════════════════════════════════════════════════════════════

def run_bayesian_config(
    benchmark: str = "gsm8k",
    n_samples: int = 30,
    n_iterations: int = 30,
    n_candidates_per_iter: int = 40,
    seed: int = 42,
):
    """Run Bayesian Optimization over agent config space."""
    from evaluate import load_gsm8k, load_humaneval

    if benchmark == "gsm8k":
        samples = load_gsm8k(n=n_samples, seed=seed)
    elif benchmark == "humaneval":
        samples = load_humaneval(n=n_samples, seed=seed)
    else:
        raise ValueError(f"Unknown: {benchmark}")

    print(f"═══ Bayesian-Config ═══")
    print(f"Benchmark: {benchmark}, Samples: {len(samples)}, Iterations: {n_iterations}")
    print(f"Candidates per iteration: {n_candidates_per_iter}")
    print()

    gp = SimpleGP(length_scale=0.5, noise=0.1)
    observed_X = []
    observed_y = []
    observed_genomes = []
    best_score = 0.0
    best_genome = None

    def _eval_genome(genome):
        reset_cost_tracking()
        result = fast_eval(genome, samples, benchmark)
        cost = get_session_cost()
        return genome, result, cost

    # Phase 1: Initial random exploration (5 diverse points)
    print("── Phase 1: Initial Exploration ──")
    init_genomes = [
        Genome(name="bo_direct", model=CHEAP, stages=[
            Stage(action="generate", temperature=0.0, condition="always",
                  system_prompt="Think step by step. Answer after ####."),
        ]),
        Genome(name="bo_code", model=CHEAP, stages=[
            Stage(action="generate_code", temperature=0.0, condition="always"),
        ]),
        Genome(name="bo_ensemble3", model=CHEAP, stages=[
            Stage(action="generate", temperature=0.0, condition="always",
                  system_prompt="Think step by step. Answer after ####."),
            Stage(action="generate", temperature=0.3, condition="always",
                  system_prompt="Solve carefully. Answer after ####."),
            Stage(action="generate_code", temperature=0.0, condition="always"),
            Stage(action="vote", condition="always"),
        ]),
        Genome(name="bo_verify", model=CHEAP, stages=[
            Stage(action="generate", temperature=0.0, condition="always",
                  system_prompt="Think step by step. Answer after ####."),
            Stage(action="verify", condition="always"),
            Stage(action="repair", condition="after_failure"),
        ]),
        Genome(name="bo_full", model=CHEAP, stages=[
            Stage(action="generate", temperature=0.0, condition="always",
                  system_prompt="Think step by step. Answer after ####."),
            Stage(action="generate_code", temperature=0.0, condition="always"),
            Stage(action="verify", condition="always"),
            Stage(action="repair", condition="after_failure"),
            Stage(action="vote", condition="always"),
        ]),
    ]

    with ThreadPoolExecutor(max_workers=len(init_genomes)) as ex:
        futures = [ex.submit(_eval_genome, g) for g in init_genomes]
        for f in as_completed(futures):
            genome, result, cost = f.result()
            features = genome_to_features(genome)
            observed_X.append(features)
            observed_y.append(result["score"])
            observed_genomes.append(genome)
            if result["score"] > best_score:
                best_score = result["score"]
                best_genome = genome
            print(f"  {genome.name:20s} | {result['score']:5.1f}%")

    print(f"Best initial: {best_score:.1f}%\n")

    # Phase 2: Bayesian Optimization loop
    print("── Phase 2: Bayesian Optimization ──")

    for iteration in range(n_iterations):
        # Fit GP to observations
        X = np.array(observed_X)
        y = np.array(observed_y)
        gp.fit(X, y)

        # Generate candidate genomes
        candidates = []

        # 1. Random exploration (30%)
        n_random = n_candidates_per_iter * 3 // 10
        for i in range(n_random):
            candidates.append(random_genome(f"bo_rand_{iteration}_{i}"))

        # 2. Perturbations of best known (40%)
        n_perturb = n_candidates_per_iter * 4 // 10
        for i in range(n_perturb):
            # Pick from top-3 observed
            top_indices = np.argsort(y)[-3:]
            parent = observed_genomes[random.choice(top_indices)]
            magnitude = random.uniform(0.2, 0.8)
            candidates.append(perturb_genome(parent, magnitude))
            candidates[-1].name = f"bo_perturb_{iteration}_{i}"

        # 3. Perturbations of best genome specifically (30%)
        n_exploit = n_candidates_per_iter - n_random - n_perturb
        for i in range(n_exploit):
            candidates.append(perturb_genome(best_genome, random.uniform(0.1, 0.5)))
            candidates[-1].name = f"bo_exploit_{iteration}_{i}"

        # Score candidates by Expected Improvement
        candidate_features = np.array([genome_to_features(g) for g in candidates])
        mu, var = gp.predict(candidate_features)
        ei = expected_improvement(mu, var, best_score)

        # Select top candidate by EI
        best_ei_idx = np.argmax(ei)
        selected = candidates[best_ei_idx]
        selected.name = f"bo_iter{iteration+1}"
        selected.model = CHEAP

        # Evaluate selected candidate
        reset_cost_tracking()
        result = fast_eval(selected, samples, benchmark)
        cost = get_session_cost()

        # Update observations
        features = genome_to_features(selected)
        observed_X.append(features)
        observed_y.append(result["score"])
        observed_genomes.append(selected)

        marker = ""
        if result["score"] > best_score:
            best_score = result["score"]
            best_genome = selected
            marker = " *** NEW BEST ***"

        if (iteration + 1) % 3 == 0 or marker:
            print(f"  [{iteration+1:3d}/{n_iterations}] score={result['score']:5.1f}% | "
                  f"EI={ei[best_ei_idx]:.4f} | "
                  f"GP_pred={mu[best_ei_idx]:.1f}±{np.sqrt(var[best_ei_idx]):.1f} | "
                  f"best={best_score:.1f}%{marker}")
            if marker:
                n_stages = len(selected.stages)
                actions = [s.action for s in selected.stages]
                print(f"         stages={n_stages}: {' → '.join(actions)}")

    # Final summary
    print(f"\n{'═'*50}")
    print(f"BAYESIAN-CONFIG COMPLETE")
    print(f"{'═'*50}")
    print(f"Best score: {best_score}%")
    print(f"Best genome ({len(best_genome.stages)} stages):")
    for i, s in enumerate(best_genome.stages):
        cond = f" [if {s.condition}]" if s.condition != "always" else ""
        print(f"  {i+1}. {s.action}(t={s.temperature}){cond}")

    print(f"\nTotal evaluations: {len(observed_y)}")
    print(f"GP observations: {len(observed_X)}")

    # Save
    with open("best_bayesian.json", "w") as f:
        json.dump(best_genome.to_dict(), f, indent=2)

    return {"score": best_score, "genome": best_genome, "n_evals": len(observed_y)}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="gsm8k")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--candidates", type=int, default=40)
    args = p.parse_args()
    run_bayesian_config(args.benchmark, args.n, args.iters, args.candidates)
