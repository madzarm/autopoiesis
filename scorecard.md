# Approach Scorecard

| Approach | Search Space | Search Algo | Quick Eval (GSM8K/30) | Quick Eval (HE/20) | Evals Done | Status |
|----------|-------------|-------------|----------------------|-------------------|------------|--------|
| genesis | linear pipeline of stages | evolutionary (LLM-guided + mutation + crossover) | 96.7% | 95.0% | 3 | parked |
| dag_evolve | DAG (nodes=ops, edges=data) | evolutionary (graph-aware: add/remove/rewire/LLM) | running... | — | 0 | exploring |
| mcts_morph | decision tree of design choices | MCTS with UCB1 | 93.3% | — | 1 | exploring |
| immune_qd | MAP-Elites archive (cost×strategy) | quality-diversity + somatic hypermutation | running... | — | 0 | exploring |
| bayesian_config | 11-dim continuous feature encoding | GP surrogate + Expected Improvement | running... | — | 0 | exploring |
