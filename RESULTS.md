# ADAS Research Results — 11 Approaches, SOTA-Competitive

## SOTA Comparison (gpt-4o-mini backbone, full test sets)

| Method | GSM8K | MATH | HumanEval | Source |
|--------|-------|------|-----------|--------|
| **AutoMaAS** | **95.4%** | 57.1% | **97.2%** | Preprint Oct 2025 |
| **Ours (AIDE)** | **94.16%** | **58.0%** | 93.29% | This work |
| AFlow | 93.5% | 56.2% | 94.7% | ICLR 2025 Oral |
| MaAS | 92.3% | 51.82% | 92.85% | ICML 2025 Oral |

**We beat MaAS on all 3 benchmarks. We beat AFlow on MATH. Competitive with AFlow on GSM8K/HumanEval.**

Our architectures:
- GSM8K: simple CoT (94.16%) — discovered by search
- MATH: 3-candidate + code_verify (58.0%) — discovered by search
- HumanEval: 5-candidate + 2-repair (93.29%) — discovered by search

## Full Test Set Results (gpt-5.4-nano backbone)

| Approach | GSM8K (1319) | HumanEval (164) | Avg |
|----------|-------------|-----------------|-----|
| DAG-Evolve (5-stage) | 90.14% | 89.63% | 89.88% |
| Adaptive-Universal (2-stage) | 82.49% | 90.24% | 86.36% |

## Cross-Benchmark Comparison (50 GSM8K + 20 HumanEval, format-neutral, fixed vote)

| Approach | GSM8K/50 | HE/20 | Avg | Stages |
|----------|---------|-------|-----|--------|
| **DAG-Evolve** | **92.0%** | 90.0% | **91.0%** | 5 |
| Immune-QD | 88.0% | 90.0% | 89.0% | 5 |
| Bayesian-Config | 88.0% | 90.0% | 89.0% | 6 |
| MCTS-Morph | 84.0% | 90.0% | 87.0% | 2 |
| baseline_cot | 78.0% | 95.0% | 86.5% | 1 |
| Genesis | 82.0% | 90.0% | 86.0% | 3 |
| LLM-Architect | 82.0% | 90.0% | 86.0% | 2 |

## 11 Approaches Implemented

1. **Genesis** — Evolutionary search over linear pipeline of stages
2. **DAG-Evolve** — Evolutionary search over DAG (graph) architectures
3. **MCTS-Morph** — Monte Carlo Tree Search over design decisions
4. **Immune-QD** — MAP-Elites quality-diversity archive
5. **Bayesian-Config** — Gaussian Process surrogate + Expected Improvement
6. **LLM-Architect** — Strong LLM as direct search algorithm
7. **Hybrid-MCTS-Evo** — Two-level: MCTS for structure + evolution for parameters
8. **Adaptive-Universal** — Multi-benchmark simultaneous optimization
9. **Meta-Ensemble** — Router over discovered agents from all approaches
10. **Evo-Devo** — Evolving developmental PROGRAMS that generate architectures
11. **AutoFlow-Converge** — Self-directing multi-turn agent

## Key Findings

1. **Vote primitive must be task-aware**: Fixed vote from extracting numbers only → detecting code vs math. This changed 0% HumanEval → 90% for 4 approaches.
2. **Complex pipelines beat simple when vote works**: DAG-Evolve (5-stage) beats MCTS-Morph (2-stage) by 4% cross-benchmark.
3. **Multi-benchmark optimization prevents format-specificity**: Single-benchmark search finds format-dependent heuristics.
4. **LLM-as-search converges fastest**: 3 iterations to find competitive designs vs 15 generations for evolution.
5. **Evaluation methodology matters**: AST-based code extraction, SymPy symbolic comparison, and retry logic can account for 3-5% score difference.

## Evaluation Methodology (matching AFlow/MaAS)
- GSM8K: #### extraction → \boxed{} → last number, tolerance 1e-6
- HumanEval: AST-based code sanitization, 15s execution timeout
- MATH: 3-tier comparison (string, numeric 1e-3, SymPy symbolic)
