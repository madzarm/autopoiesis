# Ideas Backlog

## Key Research Insights
- **Format-specificity trap**: vote/verify primitives extract numbers (####), breaking on HumanEval. 4/6 approaches scored 0% on HumanEval.
- **Simplicity wins**: 2-stage designs (generate+code, generate+generate) match or beat 5-6 stage pipelines
- **96.7% appears to be the ceiling for GSM8K/30 with gpt-5.4-nano**
- **Multi-benchmark optimization is critical**: single-benchmark optimization leads to format-dependent designs

## High Priority (novel + feasible)
- [ ] Task-adaptive primitives — vote/verify that detect task type and adjust extraction strategy
- [ ] Pareto-front search — explicitly optimize along accuracy/cost/generalization dimensions
- [ ] Curriculum learning — evolve on easy benchmarks first, then adapt to harder ones
- [ ] Self-play validation — agents evaluate each other's outputs, not just extract-and-compare

## Medium Priority (interesting but unclear)
- [ ] Swarm rules — evolve local interaction rules between micro-agents
- [ ] Morphogenetic positional encoding — agent role determined by graph position
- [ ] Co-evolutionary species — multiple agent types evolve in separate populations
- [ ] Hypergraph agents — higher-order connections between agent groups
- [ ] Program synthesis — search over architecture-GENERATING programs

## Low Priority / Long-shot
- [ ] CMA-ES over continuous config embeddings
- [ ] Neural Darwinism / pruning — start massive, prune by co-success
- [ ] Auction-based routing — market dynamics allocate compute
- [ ] RL controller — train a policy that generates architectures

## Completed
- [x] Genesis evo-devo pipelines — 96.7% GSM8K/30, 95% HE/20. Parked.
- [x] DAG-Evolve — 96.7% GSM8K/30. Graph topology matches linear pipeline performance.
- [x] MCTS-Morph — 93.3% GSM8K/30. UCB1 search over design decisions.
- [x] Immune-QD — 93.3% GSM8K/30. MAP-Elites archive, routing didn't help.
- [x] Bayesian-Config — 93.3% GSM8K/30. GP+EI works after normalization fix.
- [x] LLM-Architect — 96.7% GSM8K/30, best cross-benchmark (85.5% avg).
- [x] Hybrid-MCTS-Evo — 96.7% GSM8K/30. Two-level search converges fast.
- [x] Adaptive-Universal — 90.0% avg (86.7% GSM + 93.3% HE). Best multi-bench!
