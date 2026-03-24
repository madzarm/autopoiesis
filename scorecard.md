# Approach Scorecard

## Cross-Benchmark Comparison (GSM8K/50 + HumanEval/20) — with fixed vote

| Approach | Search Space | Search Algo | GSM8K/50 | HE/20 | **Avg** | Stages | Status |
|----------|-------------|-------------|----------|-------|---------|--------|--------|
| **dag_evolve** | DAG graph | graph evolution | **92.0%** | 90.0% | **91.0%** | 5 | **winner** |
| immune_qd | MAP-Elites grid | quality-diversity | 88.0% | 90.0% | 89.0% | 5 | strong |
| bayesian_config | 11-dim features | GP+EI | 88.0% | 90.0% | 89.0% | 6 | strong |
| adaptive_universal | configs | multi-bench search | 90.0%* | 93.3%* | 91.7%* | 2 | *math prompts |
| mcts_morph | decision tree | MCTS/UCB1 | 84.0% | 90.0% | 87.0% | 2 | good |
| baseline_cot | — | — | 78.0% | 95.0% | 86.5% | 1 | baseline |
| genesis | linear pipeline | evolutionary | 82.0% | 90.0% | 86.0% | 3 | good |
| llm_architect | configs | LLM-as-search | 82.0% | 90.0% | 86.0% | 2 | good |
| baseline_code | — | — | 84.0% | 55.0% | 69.5% | 1 | baseline |

*Adaptive-Universal used math-specific prompts for GSM8K, giving it an unfair advantage. Numbers not directly comparable.

## Full Test Set Validation
- Adaptive-Universal: GSM8K/1319 = 82.9%, HumanEval/164 = 90.2%

## Key Finding
The 0% HumanEval scores from the first comparison were caused by a format bug in `prim_vote` (extracted numbers, destroying code). After fixing vote to be task-aware, ALL approaches work cross-benchmark. Complex pipelines with verify+repair BEAT simple designs (91% vs 87%).
