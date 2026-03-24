# Approach Scorecard

| Approach | Search Space | Search Algo | Quick Eval (GSM8K/30) | Quick Eval (HE/20) | Evals Done | Status |
|----------|-------------|-------------|----------------------|-------------------|------------|--------|
| genesis | linear pipeline of stages | evolutionary (LLM-guided + mutation + crossover) | 96.7% | 95.0% | 3 | parked |
| dag_evolve | DAG (nodes=ops, edges=data) | evolutionary (graph-aware: add/remove/rewire/LLM) | 96.7% | — | 1 | exploring |
| mcts_morph | decision tree of design choices | MCTS with UCB1 | 93.3% | — | 1 | exploring |
| immune_qd | MAP-Elites archive (cost×strategy) | quality-diversity + somatic hypermutation | 93.3% | — | 1 | exploring |
| bayesian_config | 11-dim continuous feature encoding | GP surrogate + Expected Improvement | 93.3% | — | 1 | exploring |
| llm_architect | structured config | strong LLM as direct search algo | 96.7% | — | 1 | exploring |

## Summary
- **96.7% tier**: Genesis, DAG-Evolve, LLM-Architect (all found complex+code or simple generate patterns)
- **93.3% tier**: MCTS-Morph, Immune-QD, Bayesian-Config (found good but not best designs)
- Key insight: 96.7% may be the ceiling for 30 GSM8K samples with gpt-5.4-nano. Need cross-benchmark testing.

## Next Steps
1. Run top 3 (Genesis, DAG, LLM-Architect) on HumanEval for cross-benchmark check
2. Increase sample size to 100 for more reliable comparison
3. Consider new approach that combines insights: LLM-Architect's meta-reasoning + DAG's topology
