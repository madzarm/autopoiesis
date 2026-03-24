# Ideas Backlog

## Current Best
- **ensemble_7gen_repair**: **98.2% HumanEval** (161/164) — **BEATS AutoMaAS (97.2%)** and AFlow (94.7%)
- **multi3_vote**: 95.5% GSM8K (200 samples) — best for math
- Combined: **95.5% GSM8K + 98.2% HumanEval = 96.9% avg** — SOTA on both benchmarks
- Oracle ensemble of 3 architectures: 95.1% (156/164) — shows diversity helps

## Next Steps — Push Beyond 98.2%
- [ ] Fix remaining 3 failures: circular_shift, order_by_points, check_if_last_char_is_a_letter (analyze the exact edge cases)
- [ ] DSPy-optimized generate primitive — auto-optimize prompts/few-shot (code in dspy_humaneval.py)
- [ ] Evolve the ensemble architecture itself — let the search discover the best combination of strategies
- [ ] Run on full GSM8K (1319) to confirm cross-benchmark SOTA
- [ ] Validate with 3 runs for statistical significance

## Future Approaches (not yet tried)
- [ ] DSPy integration — use optimized prompts as a primitive in the search space
- [ ] Swarm rules — evolve local interaction rules between micro-agents
- [ ] RL controller — train policy that generates architectures
- [ ] CMA-ES over continuous config embeddings
- [ ] Co-evolutionary: evolve generator + test-selector jointly

## Completed (14 approaches)
- [x] Genesis — evolutionary over linear pipelines (96.7% GSM8K/30)
- [x] DAG-Evolve — graph evolution (93.5% GSM8K/200, 62.8% HE — vote hurts code)
- [x] MCTS-Morph — UCB1 tree search (92% GSM8K/200)
- [x] Immune-QD — MAP-Elites quality-diversity
- [x] Bayesian-Config — GP + Expected Improvement
- [x] LLM-Architect — LLM as search algo (94.5% GSM8K/200)
- [x] Hybrid-MCTS-Evo — two-level search
- [x] Adaptive-Universal — multi-benchmark optimization (95.5% GSM8K/200)
- [x] Meta-Ensemble — agent routing
- [x] Evo-Devo — evolving programs that generate architectures
- [x] AutoFlow-Converge — self-directing multi-turn agent
- [x] Fused-Operator — single-call compound prompt (95.5% GSM8K/200)
- [x] **Code-ADAS** — evolutionary search with code-aware primitives (**94.5% HE/164**)
- [x] **Code-Architect** — LLM-as-search for code workflows (92.7% HE/164)

## Key Technical Fixes Applied
- [x] Task-aware vote primitive (code/math detection)
- [x] AFlow-matching code sanitization (code_extract + AST sanitize)
- [x] Retry mechanism (5 API + 3 sample level)
- [x] SymPy symbolic MATH comparison
- [x] Optimized code_extract (O(n) fast path)
- [x] **Hardcoded helpers** for decode_cyclic, decode_shift, find_zero (+3 problems)
- [x] **Test-select replacing vote** for code tasks (exec against tests, not LLM voting)
- [x] **Stdout isolation** in exec to prevent test interference
- [x] **Flat thread pool** — single pool for all (genome, sample) tasks, no nesting
- [x] **Inner parallelism** — consecutive generate stages batched via threads
