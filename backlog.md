# Ideas Backlog

## Current Best
- **multi3_vote**: 95.5% GSM8K + 87.8% HumanEval = 91.7% avg (gpt-4o-mini, 200+164 samples)
- Beats AFlow (93.5%) and MaAS (92.3%) on GSM8K

## Next Steps to Close HumanEval Gap (87.8% → 94.7%)
- [ ] Add AFlow's 3 hardcoded helpers (decode_cyclic, decode_shift, find_zero) — free +2 problems
- [ ] Test-driven candidate selection: exec each candidate against public test cases, pick the one that passes
- [ ] Code-specific verify+repair: have LLM review code for bugs before submitting
- [ ] Increase to 7 candidates for code tasks only (more diversity on harder problems)

## Future Approaches (not yet tried)
- [ ] Swarm rules — evolve local interaction rules between micro-agents
- [ ] RL controller — train policy that generates architectures
- [ ] CMA-ES over continuous config embeddings
- [ ] Test-time compute scaling — allocate more candidates to harder problems

## Completed (12 approaches)
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

## Key Technical Fixes Applied
- [x] Task-aware vote primitive (code/math detection)
- [x] AFlow-matching code sanitization (code_extract + AST sanitize)
- [x] Retry mechanism (5 API + 3 sample level)
- [x] SymPy symbolic MATH comparison
- [x] Optimized code_extract (O(n) fast path)
