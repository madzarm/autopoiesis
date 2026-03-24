# Approach Scorecard

| Approach | Search Space | Search Algo | GSM8K/30 | HE/20 | Cross-bench Avg | Evals | Status |
|----------|-------------|-------------|----------|-------|-----------------|-------|--------|
| genesis | linear pipeline | evolutionary | 96.7% | 95.0% | 43.0%* | 3 | parked |
| dag_evolve | DAG graph | graph evolution | 96.7% | — | 46.0%* | 1 | parked |
| mcts_morph | decision tree | MCTS/UCB1 | 93.3% | — | 74.5% | 1 | parked |
| immune_qd | MAP-Elites grid | quality-diversity | 93.3% | — | 46.0%* | 1 | parked |
| bayesian_config | 11-dim features | GP+EI | 93.3% | — | 46.0%* | 1 | parked |
| llm_architect | configs | LLM-as-search | 96.7% | — | 85.5% | 1 | parked |
| hybrid_mcts_evo | two-level | MCTS+evo | 96.7% | — | — | 1 | parked |
| **adaptive_universal** | configs | **multi-bench** | 90.0% | 93.3% | **91.7%** | 1 | **winner** |
| meta_ensemble | agent router | portfolio | — | — | running | 0 | exploring |

*0% HumanEval due to format-specific vote primitive

## Key Insight
Multi-benchmark optimization (Adaptive-Universal) is the only approach that avoids the format-specificity trap and achieves strong cross-benchmark generalization.
