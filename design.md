# Design Document — Multiple ADAS Approaches

## Approach 1: Genesis (DONE — parked)
- **Search space**: Linear pipeline of conditional stages
- **Search algo**: Evolutionary (LLM-guided mutation + crossover + random)
- **Novelty**: Phenotypic plasticity — same genome, different behavior per problem
- **Result**: 96.7% GSM8K/30, 95.0% HumanEval/20
- **Limitation**: Linear pipeline can't express branching/merging workflows

## Approach 2: DAG-Evolve — Directed Acyclic Graph Agent Networks
- **Search space**: DAG where nodes are agent operations and edges are data flows
- **Search algo**: Evolutionary with graph-aware operators (node add/remove, edge rewire, subgraph swap)
- **Novelty**: Agents can BRANCH (fork computation into parallel paths) and MERGE (combine results). This is strictly more expressive than linear pipelines — it can represent pipelines as a special case, plus fan-out/fan-in patterns, conditional routing, and feedback loops (via unrolling).
- **Key difference from AFlow**: AFlow uses MCTS over code that defines edges. We use evolutionary search over explicit graph structures with typed ports. The graph IS the search space, not code that generates graphs.
- **Key difference from Genesis**: Genesis stages execute sequentially with conditional skipping. DAG-Evolve has true parallel branches and merge operations.
- **Hypothesized advantage**: Can discover parallel ensemble patterns (fan-out → diverse generate → merge → vote) that pipelines can't naturally express.
- **Risk**: Larger search space = slower convergence. Graph operations are more complex than list operations.

## Approach 3: MCTS-Morph — Monte Carlo Tree Search over Morphological Space
- **Search space**: Tree of agent "body plans" — each node in the MCTS tree represents a partial agent design, and children are refinements/extensions
- **Search algo**: MCTS with UCB1 selection + LLM expansion + rollout evaluation
- **Novelty**: Unlike evolutionary search (which maintains a population and breeds), MCTS builds a TREE of designs where each path from root to leaf is a sequence of design decisions. This naturally handles the explore/exploit tradeoff via UCB1, and the tree structure provides memory of what was tried.
- **Key difference from AFlow**: AFlow's MCTS nodes are complete workflows and modifications are expansions. Our MCTS nodes are PARTIAL designs — the tree represents the space of design DECISIONS (add CoT? add verification? add code execution?), not the space of complete designs. Leaf nodes are complete agents.
- **Key difference from Genesis**: Genesis evolves a flat population. MCTS-Morph grows a tree of design decisions, reusing what worked in similar branches.
- **Hypothesized advantage**: Better exploration via UCB1. Natural pruning of bad branches. Design decisions are compositional — "add verification" works across different base designs.
- **Risk**: MCTS rollouts (random completions) may be too noisy for agent design.

## Approach 4: Immune-QD — Quality-Diversity Repertoire (IMPLEMENTED)
- **Search space**: MAP-Elites archive indexed by (cost_bin, strategy_type) — 15 niches
- **Search algo**: Quality-Diversity with clonal selection, somatic hypermutation (error-diagnosis-driven targeted mutation), niche targeting, and crossover
- **Novelty**: Instead of finding ONE best agent, maintain a REPERTOIRE of diverse specialists. At inference time, route each problem to the best-matching specialist based on problem features. Biological analogy: immune repertoire with antigen-antibody matching.
- **Key difference from Genesis/DAG/MCTS**: Optimizes for DIVERSITY, not just best score. The archive IS the solution — a portfolio of specialists, not a single champion.
- **Hypothesized advantage**: Robust to problem diversity. Different problems get different agents. Routing can outperform any single best agent.
- **Risk**: Routing heuristic may be too simple. Niche granularity may be wrong.

## Approach 5: Bayesian-Config — GP Surrogate Optimization (IMPLEMENTED)
- **Search space**: Continuous feature encoding of agent configs (11-dimensional)
- **Search algo**: Bayesian Optimization with Gaussian Process surrogate + Expected Improvement acquisition
- **Novelty**: Pure function optimization — no population, no tree, no archive. Each evaluation updates a global probabilistic model. Most data-efficient approach. Completely different paradigm from evolutionary/MCTS/QD.
- **Key difference from all others**: Mathematical — uses a statistical surrogate model to predict which configs will score well, then evaluates the most promising. AutoML-inspired.
- **Hypothesized advantage**: Data-efficient (good with few evaluations). Global model of config space. Natural uncertainty quantification via GP variance.
- **Risk**: Feature encoding may lose important information. GP may not capture complex config interactions. 11-dim feature space may be too coarse.

## Approach 6: LLM-Architect — Strong LLM as Direct Search Algorithm (IMPLEMENTED)
- **Search space**: Structured configs proposed by strong LLM
- **Search algo**: LLM proposes designs based on error analysis + history
- **Novelty**: No population, no GP, no tree. The LLM IS the search algorithm. Uses a "design journal" of accumulated insights. The LLM reasons about WHY things fail and proposes targeted fixes.
- **Result**: 96.7% GSM8K/30. Best cross-benchmark: 96% GSM8K/50 + 75% HumanEval/20 = 85.5% avg
- **Key finding**: Simple 2-stage design (generate + conditional generate) outperforms complex pipelines cross-benchmark because it's format-agnostic.

## Approach 7: Hybrid-MCTS-Evo — Tree Search for Structure + Evolution for Parameters (IMPLEMENTED)
- **Search space**: Two-level: MCTS over structural decisions, mini-evolution for parameters
- **Search algo**: Outer MCTS with UCB1 + Inner evolutionary parameter optimization
- **Novelty**: Addresses MCTS-Morph's weakness (noisy rollouts) by replacing random parameter selection with mini-evolution. Addresses Genesis's weakness (wastes time on bad structures) by using MCTS to guide structural exploration.
- **Result**: 96.7% GSM8K/30 with single generate_code stage.

## Approach 8: Adaptive-Universal — Multi-Benchmark Generalization Search (IMPLEMENTED)
- **Search space**: Hybrid LLM-Architect + Evolution with multi-benchmark fitness
- **Search algo**: Multi-objective: optimize GSM8K + HumanEval simultaneously
- **Novelty**: First approach to optimize for GENERALIZATION across benchmarks. Avoids format-specific primitives (vote) that break on non-math tasks. Task-adaptive primitives.
- **Key insight from comparison**: Most GSM8K-optimized designs fail on HumanEval (0%) because vote/verify are format-dependent.

## Previous (pre-refactor)
AIDE: Adaptive Immune-inspired Design Evolution

## One-Paragraph Summary

AIDE (Adaptive Immune-inspired Design Evolution) is a novel ADAS method that draws from the adaptive immune system to automatically discover high-performing agent configurations. Like the immune system maintains a diverse repertoire of lymphocytes and rapidly amplifies those that match a threat (clonal selection), AIDE maintains a diverse population of agent configs and selectively amplifies those that perform well. It uses **execution-trace-conditioned mutation** (analogous to somatic hypermutation guided by antigen exposure), **affinity maturation** (iterative refinement of promising configs based on error analysis), and **immune memory** (an archive of proven solutions that provides rapid responses to similar future tasks). Unlike AFlow's MCTS or ADAS's open-ended code generation, AIDE operates in a bounded configuration space (like EvoMAS) but adds three key innovations: (1) error-diagnosis-driven mutation where an LLM analyzes WHY an agent failed and proposes targeted fixes, (2) repertoire diversity enforcement via a quality-diversity archive, and (3) cost-aware Pareto selection.

## Key Innovations (vs. prior work)

1. **Error-Diagnosis-Driven Mutation (Somatic Hypermutation)**: Unlike random mutations (EvoAgent) or blind LLM proposals (ADAS), AIDE feeds the meta-agent specific failure examples with error analysis. The meta-agent sees WHAT went wrong and proposes targeted config changes — analogous to how somatic hypermutation focuses on the antigen-binding region. This makes mutations much more sample-efficient.

2. **Quality-Diversity Archive (Immune Repertoire)**: Instead of just keeping the best agent (clonal selection), AIDE maintains a diverse archive organized by behavioral niches. Two agents that achieve similar scores but through different strategies (e.g., one uses decomposition, another uses debate) are both kept — ensuring the system doesn't collapse to a single strategy. This is inspired by how the immune system maintains diversity to handle novel pathogens.

3. **Multi-Phase Search with Adaptive Focus**:
   - Phase 1 (Innate): Broad random exploration of the search space
   - Phase 2 (Adaptive): LLM-guided refinement of promising regions
   - Phase 3 (Memory): Retrieval and adaptation of proven configs for new benchmarks
   This mirrors innate → adaptive → memory immune responses.

4. **Cost-Aware Pareto Selection**: Every config is evaluated on both accuracy AND cost. Selection favors configs on the Pareto frontier — high accuracy for their cost class. This prevents the system from only finding expensive solutions.

5. **Cross-Benchmark Transfer**: Configs that work well on one benchmark are automatically tested on others, enabling discovery of generally capable agents rather than benchmark-specific ones.

## Search Space Definition

The search space is a structured configuration:

```python
{
    "reasoning": ["direct", "cot", "cot_sc", "decompose", "analogy", "abstract"],
    "planning": ["none", "step_by_step", "recursive", "divide_conquer"],
    "reflection": ["none", "self_check", "self_refine", "critic"],
    "ensemble": ["none", "majority_vote", "best_of_n", "debate"],
    "ensemble_n": [1, 3, 5, 7],
    "output_format": ["free", "structured", "step_numbered"],
    "temperature": [0.0, 0.1, 0.3, 0.5, 0.7, 1.0],
    "reflection_rounds": [1, 2, 3],
    "persona": str,  # Free-form system prompt
    "custom_instructions": str,  # Free-form additional instructions
}
```

Total combinatorial space (excluding free-form strings): ~10,000 discrete configs.
With persona/instructions, the space is effectively infinite but structured.

## Search Algorithm Pseudocode

```
INITIALIZE:
  archive = []
  population = [random_config() for _ in range(INIT_POP)]

  # Phase 1: Innate — broad exploration
  for config in population:
    score, details = evaluate(config, benchmark, n_samples)
    add_to_archive(config, score, details)

LOOP:
  # Select parent(s) from archive
  if len(archive) < EXPLORE_THRESHOLD:
    # Still exploring — use random + LLM proposals
    parent = select_from_archive(strategy="diverse")
    child = llm_propose(archive_summary, error_examples)
  else:
    # Exploit — refine best configs
    parent = select_from_archive(strategy="pareto_best")
    errors = get_error_examples(parent)
    child = llm_mutate(parent, errors)  # Error-diagnosis-driven

  # Evaluate child
  score, details = evaluate(child, benchmark, n_samples)

  # Archive update (quality-diversity)
  niche = compute_niche(child.config)
  if is_pareto_improvement(child, archive[niche]):
    archive[niche] = child

  # Cross-benchmark transfer (every K steps)
  if step % K == 0:
    best = get_pareto_front(archive)
    for config in best:
      for other_benchmark in benchmarks:
        evaluate(config, other_benchmark)

  # Adaptive focus
  if stuck_count > STUCK_THRESHOLD:
    # Radical exploration — try something very different
    child = random_config() or crossover(diverse_parents)
```

## Evaluation Strategy

- Primary: GSM8K (50 samples for fast iteration, 200 for validation)
- Secondary: MGSM, DROP
- Metrics: accuracy (math), F1 (DROP), cost per sample
- Multi-objective: score vs cost Pareto frontier

## Hypothesized Advantages

1. **Sample efficiency**: Error-diagnosis mutations should find improvements faster than random mutations
2. **Diversity**: Quality-diversity archive prevents premature convergence
3. **Cost-effectiveness**: Pareto selection naturally finds cheap-but-good configs
4. **Generalization**: Cross-benchmark transfer prevents overfitting to one task
5. **Scalability**: Bounded config space is much more robust than open-ended code generation

## Computational Budget

- Meta-agent calls: gpt-4.1 (strong model) — ~$0.01-0.05 per proposal
- Inner agent evaluations: gpt-4.1-nano — ~$0.002-0.02 per eval
- Target: <$1 per experiment cycle
- Total budget: ~$50-100 for full research program
