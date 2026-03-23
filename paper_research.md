# Research Notes: AgentSquare & MaAS

## Paper 1: AgentSquare (ICLR 2025)

**Title**: AgentSquare: Automatic LLM Agent Search in Modular Design Space
**Authors**: Shang et al. (Tsinghua University)
**Venue**: ICLR 2025
**arXiv**: 2410.06153
**Code**: https://github.com/tsinghua-fib-lab/AgentSquare

### Problem Formulation
Introduces **Modularized LLM Agent Search (MoLAS)** — abstracts existing LLM agent designs into four fundamental modules with uniform I/O interfaces.

### Search Space / Architecture Representation
Four module categories with standardized I/O:

1. **Planning** — Decomposes tasks into sub-tasks. Input: task description + optional feedback. Output: sub-task sequences.
   - Variants include: IO (no planning), Sequential decomposition, TD (Task Decomposition), Voyager-style, DEPS-style, IR (discovered)

2. **Reasoning** — Solves sub-tasks via advanced prompting.
   - Variants include: CoT, CoT-SC, ToT, Self-Refine, Step Back, Thought Propagation (TP), SF-ToT (discovered), HTSS (discovered), CASRC (discovered)

3. **Tool Use** — Selects appropriate tools from predefined pool.
   - Variants include: No Tool, ToolBF (ToolBench), TH (discovered), HuggingGPT-style

4. **Memory** — Manages read/write of observations and experiences.
   - Variants include: No Memory, Dilu-style, Generative Agents-style, Hierarchical (Hier), Task-specific

**Total design space**: 1050 possible combinations from 16 popular seed agents.

16 seed agents analyzed: CoT, CoT-SC, Self-refine, ToT, Step Back, Thought Propagation, HuggingGPT, Voyager, Generative Agents, DEPS, OPENAGI, Dilu, and others.

### Search / Optimization Algorithm
Two core mechanisms:

1. **Module Evolution**: Evolutionary meta-prompt generates new modules through prompt-level optimization. Uses task descriptions, existing modules, and prior performance data to create improved module variants.

2. **Module Recombination**: LLM-based proposer identifies promising module combinations by strategically replacing individual modules. Builds on current best-performing agents by swapping modules from the pool.

**Performance Predictor**: In-context surrogate model that evaluates proposed agents without full environment testing. Uses task descriptions, module profiles, and in-context performance examples. Cost is only ~0.025% of a full GPT-4o evaluation.

**Search Process**:
- Initialize experience pool with seed agents
- Iterate K episodes (terminate after 5 consecutive iterations without improvement)
- Each episode: evolution phase → recombination phase
- Performance predictor screens unpromising candidates
- GPT-4o: 9-18 iterations per task
- GPT-3.5: 8-23 iterations per task

### LLM Backbones
- GPT-4o
- GPT-3.5-turbo-0125

### Benchmarks Used
Six benchmarks across four domains:
- **Web**: WebShop, TravelPlanner
- **Embodied**: ALFWorld, ScienceWorld
- **Tool Use**: M3Tool
- **Game**: PDDL

All use success rate (0-1 scale) as metric.

### Main Results — Table 1 (GPT-4o)

#### Hand-Crafted Agents

| Method | Webshop | ALFWorld | SciWorld | M3Tool | TravelPlanner | PDDL |
|--------|---------|----------|----------|--------|---------------|------|
| CoT | 0.485 | 0.405 | 0.697 | 0.448 | 0.487 | 0.542 |
| CoT-SC | 0.512 | 0.426 | 0.656 | 0.461 | 0.413 | 0.495 |
| Self-refine | 0.461 | 0.567 | 0.654 | 0.442 | 0.000 | 0.514 |
| ToT | 0.501 | 0.437 | 0.741 | 0.453 | 0.380 | 0.476 |
| Step Back | 0.468 | 0.279 | 0.220 | 0.434 | 0.000 | 0.486 |
| TP | 0.398 | 0.404 | 0.576 | 0.387 | 0.430 | 0.518 |
| HuggingGPT | 0.519 | 0.481 | 0.680 | 0.354 | 0.510 | 0.584 |
| Voyager | 0.366 | 0.425 | 0.776 | 0.247 | 0.523 | 0.412 |
| Generative Agents | 0.499 | 0.477 | 0.663 | 0.402 | 0.480 | 0.553 |
| DEPS | 0.481 | 0.459 | 0.740 | 0.278 | 0.540 | 0.591 |
| OPENAGI | 0.506 | 0.510 | 0.718 | 0.322 | 0.533 | 0.616 |
| Dilu | 0.451 | 0.433 | 0.682 | 0.475 | 0.360 | 0.463 |

#### Search-Based Methods

| Method | Webshop | ALFWorld | SciWorld | M3Tool | TravelPlanner | PDDL |
|--------|---------|----------|----------|--------|---------------|------|
| Random | 0.533 | 0.620 | 0.704 | 0.438 | 0.563 | 0.660 |
| Bayesian | 0.549 | 0.634 | 0.749 | 0.502 | 0.537 | 0.650 |
| OPRO | 0.505 | 0.380 | 0.569 | 0.309 | 0.523 | 0.589 |
| ADAS | 0.521 | 0.543 | 0.754 | 0.475 | 0.373 | 0.568 |
| **AgentSquare** | **0.607** | **0.695** | **0.781** | **0.524** | **0.583** | **0.669** |

**Average performance gain over best hand-crafted agent: 17.2%**

Per-task gains: Webshop +14.1%, ALFWorld +26.1%, SciWorld +20.5%, M3Tool +30.6%, TravelPlanner +6.0%, PDDL +6.0%

### GPT-3.5-turbo Results (Table A.3)

| Method | Webshop | ALFWorld | SciWorld | M3Tool | TravelPlanner | PDDL |
|--------|---------|----------|----------|--------|---------------|------|
| AgentSquare | 0.617 | 0.651 | 0.432 | 0.285 | 0.520 | 0.219 |

### Ablation Study (Table 2 — GPT-4o)

| Configuration | Webshop | ALFWorld | SciWorld | M3Tool | TravelPlanner | PDDL |
|---------------|---------|----------|----------|--------|---------------|------|
| Full AgentSquare | 0.607 | 0.695 | 0.781 | 0.524 | 0.583 | 0.669 |
| w/o module evolution | 0.564 | 0.649 | 0.736 | 0.502 | 0.577 | 0.614 |
| w/o module recombination | 0.560 | 0.616 | 0.710 | 0.481 | 0.280 | 0.669 |

### Ablation Study (Table A.5 — GPT-3.5)

| Configuration | Webshop | ALFWorld | SciWorld | M3Tool | TravelPlanner | PDDL |
|---------------|---------|----------|----------|--------|---------------|------|
| Full AgentSquare | 0.617 | 0.651 | 0.432 | 0.285 | 0.520 | 0.219 |
| w/o evolution | 0.595 | 0.623 | 0.288 | 0.236 | 0.483 | 0.202 |
| w/o recombination | 0.578 | 0.546 | 0.310 | 0.258 | 0.267 | 0.173 |

### Discovered Best Agents (Table A.4 — module compositions)

| Task | Planning | Reasoning | Tool Use | Memory |
|------|----------|-----------|----------|--------|
| Webshop | IO | HTSS | / | Dilu |
| ALFWorld | TD | SF-ToT | / | Generative Agents |
| SciWorld | Voyager | CoT | / | Hier |
| M3Tool | / | CoT-SC | ToolBF | / |
| TravelPlanner | DEPS | CoT | TH | / |
| PDDL | IR | CASRC | / | Generative Agents |

### Search Cost
- GPT-4o: $10.51-$42.14 per task per iteration, 9-18 iterations
- GPT-3.5: $1.84-$4.25 per task per iteration, 8-23 iterations
- Performance predictor cost: ~0.025% of full evaluation

### Key Innovations
1. First to formalize Modularized LLM Agent Search (MoLAS) problem
2. Modular design space with uniform I/O interfaces enabling mix-and-match
3. Dual search mechanism: module evolution (prompt-level) + recombination (module-level)
4. In-context surrogate performance predictor for cheap candidate screening
5. Can discover novel module variants (HTSS, SF-ToT, CASRC, etc.)
6. Generates interpretable design insights

### Limitations
- Single-agent search only (no multi-agent collaboration)
- Limited to 4 fixed module categories
- Search requires multiple expensive LLM evaluations
- Design space seeded from existing human designs
- Benchmarks are relatively constrained

---

## Paper 2: MaAS (ICML 2025 Oral/Spotlight)

**Title**: Multi-agent Architecture Search via Agentic Supernet
**Authors**: Guibin Zhang, Luyang Niu, Junfeng Fang, Kun Wang, Lei Bai, Xiang Wang
**Venue**: ICML 2025 Oral (Top ~1% of 12,107 submissions)
**arXiv**: 2502.04180
**Code**: https://github.com/bingreeky/MaAS

### Problem Formulation
Shifts from finding a static one-size-fits-all agentic system to optimizing an **agentic supernet** — a probabilistic and continuous distribution of agentic architectures. Samples query-dependent multi-agent systems for each input.

### Search Space / Architecture Representation
**Agentic Supernet**: A cascaded multi-layer workflow.

**Structure**:
- **Layers**: L=4 discrete layers
- **Operators per layer**: Multiple agentic operators with parameterized probability distributions
- **Sampling**: Threshold-based with K=4 sampling times and activation threshold of 0.3

**Agentic Operators** (composite LLM-agent invocation processes):
- **Generate** — Basic text/code generation with single LLM call
- **GenerateCoT** — Chain-of-thought reasoning with exemplars
- **MultiGenerateCoT** — Parallel diversity-oriented CoT producing 3 solutions
- **ScEnsemble** — Self-consistency ensemble selecting most consistent answer
- **SelfRefine** — Error analysis and solution improvement
- **EarlyStop** — Execution termination placeholder (enables early exit for easy queries)
- **Task-specific**: CustomCodeGenerate, Test (HumanEval), Programmer (MATH/GSM8K)

**Key design**: Each operator involves m LLM-agents and n tool calls. E.g., CoT has m=1, n=0; Multi-agent debate has multiple LLM-agents with multi-turn calls.

### Search / Optimization Algorithm
**Controller Network**: Samples architectures conditioned on input queries using MoE-style mechanism.
- Sequentially constructs execution topology layer-by-layer
- Uses autoregressive factorization — P(topology) factorized across layers
- Per-operator activation scores; operators selected via cumulative confidence threshold τ

**Joint Optimization**:
1. **Supernet distribution parameters**: Updated via Monte Carlo sampling gradient approximation
2. **Agentic operators**: Updated via textual gradient estimation (environmental feedback)

**Key feature**: Query-aware early exit — easy queries exit at early layers with high probability, saving resources.

**Cost penalty**: λc=3

### LLM Backbones
- GPT-4o-mini (primary)
- Gemini-1.5-flash (transferability)
- Qwen-2-72b (transferability)
- LLaMA-3.1-70b (transferability)

### Benchmarks Used
Six benchmarks across three domains:
- **Math Reasoning**: GSM8K, MATH, MultiArith
- **Code Generation**: HumanEval, MBPP
- **Tool Use**: GAIA

### Main Results — MaAS (GPT-4o-mini)

From the MaAS paper and cross-referenced with DAAO/AutoMaAS papers:

| Method | GSM8K | MATH | HumanEval | MBPP | MultiArith | GAIA |
|--------|-------|------|-----------|------|-----------|------|
| **Single-Agent** | | | | | | |
| Vanilla | 87.45 | 46.29 | 85.71 | 72.20 | — | — |
| CoT | 87.1 | 46.4 | 88.1 | 71.8 | 96.9 | 14.7 |
| ComplexCoT | 86.9 | 46.5 | 87.5 | 72.4 | 96.7 | 14.8 |
| Self-Consistency | 87.6 | 47.9 | 88.6 | 73.6 | 96.6 | 14.9 |
| **Hand-Crafted Multi-Agent** | | | | | | |
| MultiPersona | 87.5 | 45.4 | 88.3 | 73.2 | 97.5 | 15.2 |
| LLM-Debate | 89.5 | 48.5 | 88.7 | 70.3 | 97.3 | 16.6 |
| LLM-Blender | 88.4 | 46.9 | 88.8 | 77.1 | 97.3 | 16.6 |
| DyLAN | 90.0 | 48.6 | 90.4 | 77.3 | 97.1 | 16.3 |
| AgentVerse | 89.9 | 47.4 | 89.3 | 74.3 | 97.5 | 16.3 |
| MacNet | 88.0 | 45.2 | 84.6 | 65.3 | 96.0 | 16.3 |
| **Automated Methods** | | | | | | |
| AutoAgents | 87.7 | 45.3 | 87.6 | 72.0 | 96.4 | 15.2 |
| GPTSwarm | 89.1 | 47.9 | 89.3 | 77.4 | 96.8 | 16.3 |
| ADAS | 86.1 | 43.2 | 84.2 | 68.1 | 96.0 | 16.7 |
| AgentSquare | 87.6 | 48.5 | 89.1 | 78.5 | 97.8 | 16.3 |
| AFlow | 91.2 | 51.3 | 90.9 | 81.7 | 96.2 | 18.0 |
| **MaAS** | **92.30** | **51.82** | **92.85** | **82.17** | — | — |

**Average best score across all tasks: 83.59%**
**GAIA Level 1 improvement: +18.38%**

Note: Some numbers from the AutoMaAS paper's reproduction (Table IV) which tested all baselines on all 6 benchmarks.

### MaAS Results with Different Backbones

#### GPT-4o-mini (MMLU, GSM8K, MATH, HumanEval, MBPP)
MaAS: 83.01, 92.30, 51.82, 92.85, 82.17

#### Gemini-1.5-flash
MaAS: 83.42, 92.00, 52.25, 90.55, 82.69

### GAIA Benchmark Results

| Method | Level 1 | Level 2 | Level 3 | Average |
|--------|---------|---------|---------|---------|
| GPT-4o-mini | 7.53 | 4.40 | 0 | 4.65 |
| ADAS | 13.98 | 4.40 | 0 | 6.69 |
| AFlow | 10.75 | 8.81 | 4.08 | 8.00 |
| **MaAS** | **20.45** | **18.61** | **6.25** | **17.64** |

### Cost / Efficiency Analysis (MATH benchmark)

| Method | Training Cost | Inference Cost | Total Cost | Accuracy |
|--------|--------------|----------------|------------|----------|
| AFlow | $22.50 | $1.66 | $24.16 | 51.82% |
| **MaAS** | **$3.38** | **$0.42** | **$3.80** | **51.82%** |

MaAS requires only **6-45% of the inference costs** of existing systems.
MaAS achieves **85% training cost savings** vs AFlow ($3.38 vs $22.50).

### Additional MaAS Results (from LAMaS paper comparison)

| Dataset | MaAS Accuracy | MaAS Cost | MaAS Completion Path Length |
|---------|--------------|-----------|----------------------------|
| GSM8K | 93.13% | $0.56 | 1474.6 tokens |
| HumanEval | 93.00% | $0.08 | 1810.8 tokens |
| MATH | 51.23% | $0.37 | 2218.5 tokens |

### Key Innovations
1. **Agentic Supernet concept**: Probabilistic continuous distribution over architectures (vs. discrete search)
2. **Query-dependent sampling**: Each query gets a customized multi-agent architecture
3. **Early exit mechanism**: Easy queries exit at early layers, saving resources
4. **Joint optimization**: Simultaneously optimize supernet distribution (Monte Carlo) and operators (textual gradients)
5. **Cross-backbone transferability**: Supernet transfers across different LLM backbones
6. **Dramatic cost reduction**: 6-45% of existing systems' inference costs while matching or exceeding quality
7. **NAS-inspired**: Directly applies neural architecture search (supernet/weight-sharing) paradigm to agentic systems

### Limitations
1. Text-only — no multimodal integration
2. Limited interpretability — textual gradient mechanism operates implicitly within LLM's prompt space
3. Granularity may be insufficient for very difficult inputs (noted by DAAO authors)
4. Fixed operator set (addressed by AutoMaAS follow-up)
5. 4-layer fixed depth

---

## Cross-Paper Comparison

### Design Philosophy
| Aspect | AgentSquare | MaAS |
|--------|-------------|------|
| Unit of search | Module (Planning/Reasoning/ToolUse/Memory) | Agentic operator (composite LLM workflow) |
| Architecture type | Single agent with 4 modules | Multi-agent with L=4 layers |
| Search output | One best agent per task | Distribution → query-specific agent per query |
| Adaptation | Task-level (one architecture per task) | Query-level (different architecture per query) |
| Optimization | LLM-based evolution + recombination | Gradient-based (Monte Carlo + textual gradients) |

### Benchmarks Overlap
- AgentSquare: WebShop, ALFWorld, ScienceWorld, M3Tool, TravelPlanner, PDDL
- MaAS: GSM8K, MATH, MultiArith, HumanEval, MBPP, GAIA
- **No overlapping benchmarks** — they evaluate on completely different tasks

### When Both Are Evaluated by Third Parties (AutoMaAS, GPT-4o-mini)
| Method | GSM8K | MATH | HumanEval | MBPP | MultiArith | GAIA |
|--------|-------|------|-----------|------|-----------|------|
| AgentSquare | 87.6 | 48.5 | 89.1 | 78.5 | 97.8 | 16.3 |
| AFlow | 91.2 | 51.3 | 90.9 | 81.7 | 96.2 | 18.0 |
| MaAS | 92.30 | 51.82 | 92.85 | 82.17 | — | 20.45 (L1) |

MaAS outperforms AgentSquare on the math/code benchmarks by significant margins.

### Cost Comparison
- AgentSquare search cost: $10-42 per task iteration (GPT-4o), needs 9-18 iterations
- MaAS total cost: $3.80 for MATH (training + inference)
- MaAS is dramatically cheaper at inference time

### Key Differences in Innovation
- AgentSquare: Modular decomposition + evolution of individual modules + interpretable compositions
- MaAS: Probabilistic supernet + query-dependent sampling + early exit + transferability

Both are automated agent design systems but approach the problem from fundamentally different angles. AgentSquare is more interpretable (you can see exactly which modules compose the best agent). MaAS is more efficient and adaptive (different architecture per query, early exit for easy ones).
