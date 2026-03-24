# Ideas Backlog

## High Priority (novel + feasible)
- [ ] Immune-QD — MAP-Elites quality-diversity archive of specialist agents indexed by behavioral descriptors (cost, complexity, reasoning style). Route problems to best-matching specialist at inference time. Genuinely different from evolutionary/MCTS — maintains *diversity*, not just best.
- [ ] Bayesian Opt — Gaussian process surrogate over structured config space. Data-efficient, good for expensive evaluations. Encode agent configs as continuous vectors, fit GP, acquire next config via Expected Improvement.
- [ ] Swarm rules — evolve local interaction rules between micro-agents. No global architecture, just pairwise communication rules. Emergent global behavior from simple local rules (ant colony analogy).

## Medium Priority (interesting but unclear)
- [ ] RL controller — train a small policy network that generates agent architectures token-by-token. Reward = benchmark score. REINFORCE or PPO.
- [ ] Morphogenetic positional encoding — agent role determined by position in communication graph. Same "DNA" (shared weights), different expression based on position signals.
- [ ] Co-evolutionary species — multiple agent types (planner, coder, verifier) evolve in separate populations with symbiotic fitness (team performance).
- [ ] Hypergraph agents — higher-order connections where groups of 3+ agents form a hyperegde (collective operation). Richer than pairwise DAG edges.
- [ ] Program synthesis search — search over architecture-GENERATING programs, not architectures directly. Evo-devo: genotype is a program that builds the phenotype.

## Low Priority / Long-shot
- [ ] CMA-ES over continuous config embeddings — treat discrete choices as continuous relaxation, use CMA-ES to optimize
- [ ] Neural Darwinism / pruning — start with massive over-connected agent network, prune connections based on co-success
- [ ] Microbiome co-evolution — helper agents that improve the "fitness" of primary agents, evolve symbiotically
- [ ] Auction-based routing — problems "bid" for agents, agents compete. Market dynamics allocate compute

## Tried and Parked
- [x] Genesis evo-devo pipelines — 96.7% GSM8K/30, 95% HE/20. Linear pipeline limits expressiveness.
- [ ] DAG-Evolve — running now. Graph topology more expressive than linear but search space larger.
- [ ] MCTS-Morph — running now. Decision tree search with UCB1 exploration.
