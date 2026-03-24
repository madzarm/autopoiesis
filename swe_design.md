# SWE-ADAS: Evolutionary Agent Design for SWE-bench

## Approach: Meta-Evolutionary Interactive Agent Search

### vs EvoMAS (our main competitor)

| Dimension | EvoMAS | SWE-ADAS (Ours) |
|-----------|--------|-----------------|
| Agent type | Pipeline (YAML configs) | Interactive (bash shell agent) |
| Search space | Agent count, topology, prompts, tools, models | System prompt, strategy hints, temperature, max turns, tool config |
| Mutation guidance | Raw execution traces | Structured trajectory analysis (failure type classification) |
| Evolution | Sequential: Select → Mutate → Evaluate | Population-based: MAP-Elites archive + Pareto optimization |
| Memory | Text consolidation per instance | Hierarchical: repo-level → bug-type → instance |
| Execution rate | 98%+ (YAML interpreted) | 100% (interactive agent always runs) |
| Expressiveness | Fixed pipeline (limited by config schema) | Full interactivity (agent adapts strategy live) |
| Agent backbone | Claude-3.5-Sonnet / Claude-4.5-Sonnet | Claude Sonnet (3.5 equivalent, cheaper) |

### Key Innovations

1. **Interactive Agent Evolution**: Instead of evolving static pipeline configurations,
   we evolve the system prompt and strategy of an interactive bash agent. This gives
   the agent much more flexibility to adapt its behavior based on what it discovers
   at runtime.

2. **Structured Trajectory Analysis**: EvoMAS uses raw execution traces for mutation.
   We classify failures into types (localization failure, no edit made, edit error,
   loop, timeout, empty patch) and apply targeted mutations for each type.

3. **MAP-Elites Archive**: Instead of a flat config pool, we maintain a quality-diversity
   archive indexed by (repo, difficulty). This ensures we have specialized configs for
   different problem types.

4. **Hierarchical Memory**: Repo-level patterns transfer across instances from the same
   project. Bug-type patterns transfer across repos.

### Architecture

```
MetaEvolver
├── Population (5 seed configs → grows via evolution)
│   ├── default_interactive (basic bash agent)
│   ├── methodical_explorer (phased: understand → diagnose → fix → verify)
│   ├── test_driven_fixer (find tests first, then fix)
│   ├── surgical_fixer (fast, targeted, minimal exploration)
│   └── context_builder (thorough codebase understanding)
│
├── Trajectory Analyzer
│   ├── Failure classification (localization/no_edit/loop/timeout/etc)
│   ├── Pattern extraction (repeated commands, missing steps)
│   └── Success signal extraction (what worked in good runs)
│
├── Mutation Engine (LLM-guided)
│   ├── Prompt refinement (based on failure type)
│   ├── Strategy hint injection (based on repo/bug patterns)
│   ├── Temperature adjustment
│   └── Turn budget modification
│
├── Crossover Engine
│   └── Combines system_prompt from top scorer + strategy from runner-up
│
└── Memory
    ├── Per-repo best configs
    ├── MAP-Elites archive: (repo, difficulty) → config
    └── Instance results log
```

### Evaluation

- **Primary**: SWE-bench Verified (500 instances)
- **Fast iteration**: SWE-bench Verified Mini (50 instances) or subset
- **Proxy metric**: Valid diff rate + patch plausibility
- **Real metric**: Resolved rate via sb-cli (free cloud eval)

### Cost Model

- Per instance (interactive, 25 turns): ~$0.50-0.70
- Per instance (meta-evolve, 3 candidates × 2 depths): ~$3-5
- Full Verified (500 instances, interactive only): ~$300
- Full Verified (500 instances, meta-evolve): ~$2000
- EvoMAS comparison: they don't report per-instance costs

### Target

- EvoMAS baseline (Claude-3.5-Sonnet): 63.8% on SWE-bench Verified
- Our target: >65% with interactive agent, >70% with meta-evolution
- Stretch: >75% matching EvoMAS's Claude-4.5-Sonnet result with only Sonnet

### Files

- `swe_llm.py` — Anthropic Claude client with cost tracking
- `swe_agent.py` — One-shot Agentless baseline
- `swe_interactive_agent.py` — Interactive bash agent (main agent)
- `swe_eval.py` — Evaluation harness (sb-cli + local Docker)
- `swe_evolve.py` — Evolution controller for one-shot agent
- `swe_meta_evolve.py` — Meta-evolution for interactive agent (MAIN)
- `run_swe.py` — CLI runner for all modes
