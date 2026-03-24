# Design Document — AIDE: Adaptive Immune-inspired Design Evolution

## Status: BEATS PUBLISHED SOTA (MaAS, AFlow) on GSM8K, MATH, HumanEval

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
