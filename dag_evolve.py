#!/usr/bin/env python3
"""DAG-Evolve — Directed Acyclic Graph Agent Networks.

Approach 2: Agents are DAGs where nodes are operations and edges are data flows.
Unlike Genesis (linear pipeline), DAGs can branch (fan-out) and merge (fan-in),
enabling parallel ensemble patterns and conditional routing.

Key novelty vs AFlow: We evolve explicit graph structures, not code.
Key novelty vs Genesis: True parallel branches and merge operations.
"""

import json
import re
import random
import copy
import time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm import call_llm, STRONG, MID, CHEAP, get_session_cost, reset_cost_tracking
from evaluate import extract_number, extract_math_answer, normalize_math_answer
from genesis import prim_generate, prim_generate_code, prim_verify, prim_repair, prim_vote


# ═══════════════════════════════════════════════════════════════
# DAG REPRESENTATION
# ═══════════════════════════════════════════════════════════════

@dataclass
class DAGNode:
    """A node in the agent DAG."""
    id: str
    action: str  # generate, generate_code, verify, repair, vote, merge
    temperature: float = 0.0
    system_prompt: str = ""
    # Input edges: list of node IDs whose outputs feed into this node
    inputs: list = field(default_factory=list)

    def to_dict(self):
        return {"id": self.id, "action": self.action, "temperature": self.temperature,
                "system_prompt": self.system_prompt, "inputs": self.inputs}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


@dataclass
class DAGGenome:
    """A DAG-based agent genome. Nodes connected by directed edges."""
    name: str = "unnamed"
    nodes: list = field(default_factory=list)
    output_node: str = ""  # ID of the node whose output is the final answer
    model: str = CHEAP

    def to_dict(self):
        return {"name": self.name, "model": self.model, "output_node": self.output_node,
                "nodes": [n.to_dict() for n in self.nodes]}

    @classmethod
    def from_dict(cls, d):
        nodes = [DAGNode.from_dict(n) for n in d.get("nodes", [])]
        return cls(name=d.get("name", ""), nodes=nodes,
                   output_node=d.get("output_node", ""), model=d.get("model", CHEAP))

    def get_node(self, node_id: str):
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def topo_sort(self) -> list:
        """Topological sort of nodes."""
        visited = set()
        order = []
        node_map = {n.id: n for n in self.nodes}

        def visit(nid):
            if nid in visited:
                return
            visited.add(nid)
            node = node_map.get(nid)
            if node:
                for inp in node.inputs:
                    visit(inp)
                order.append(nid)

        for n in self.nodes:
            visit(n.id)
        return order


# ═══════════════════════════════════════════════════════════════
# DAG EXECUTION — runs nodes in topological order, parallelizing independent nodes
# ═══════════════════════════════════════════════════════════════

def execute_dag(dag: DAGGenome, problem: str) -> str:
    """Execute a DAG on a problem. Independent nodes run in parallel."""
    node_map = {n.id: n for n in dag.nodes}
    results = {}  # node_id -> output string
    topo = dag.topo_sort()

    # Group nodes by "depth" (nodes at same depth can run in parallel)
    depth = {}
    for nid in topo:
        node = node_map.get(nid)
        if not node:
            continue
        if not node.inputs:
            depth[nid] = 0
        else:
            depth[nid] = max(depth.get(inp, 0) for inp in node.inputs) + 1

    max_depth = max(depth.values()) if depth else 0

    for d in range(max_depth + 1):
        nodes_at_depth = [nid for nid, dd in depth.items() if dd == d]

        def _run_node(nid):
            node = node_map[nid]
            # Gather inputs
            input_texts = [results.get(inp, "") for inp in node.inputs]
            input_combined = "\n\n".join(t for t in input_texts if t)

            if node.action == "generate":
                context = f"{problem}\n\n{input_combined}" if input_combined else problem
                r = prim_generate(context, dag.model, node.temperature, node.system_prompt)
                return nid, r["text"]

            elif node.action == "generate_code":
                context = f"{problem}\n\n{input_combined}" if input_combined else problem
                r = prim_generate_code(context, dag.model, node.temperature)
                return nid, r["text"]

            elif node.action == "verify":
                if input_combined:
                    r = prim_verify(problem, input_combined, dag.model)
                    return nid, r["feedback"]
                return nid, ""

            elif node.action == "repair":
                if input_combined:
                    r = prim_repair(problem, input_combined, "Check and fix errors.", dag.model, node.temperature)
                    return nid, r["text"]
                return nid, ""

            elif node.action in ("vote", "merge"):
                if len(input_texts) >= 2:
                    r = prim_vote(input_texts)
                    return nid, r["text"]
                return nid, input_texts[0] if input_texts else ""

            return nid, ""

        # Run nodes at this depth in parallel
        if len(nodes_at_depth) > 1:
            with ThreadPoolExecutor(max_workers=len(nodes_at_depth)) as ex:
                futures = [ex.submit(_run_node, nid) for nid in nodes_at_depth]
                for f in as_completed(futures):
                    nid, text = f.result()
                    results[nid] = text
        else:
            for nid in nodes_at_depth:
                nid, text = _run_node(nid)
                results[nid] = text

    return results.get(dag.output_node, "")


# ═══════════════════════════════════════════════════════════════
# SEED DAGs
# ═══════════════════════════════════════════════════════════════

ACTIONS = ["generate", "generate_code", "verify", "repair", "vote", "merge"]
PROMPTS = [
    "Think step by step. Answer after ####.",
    "Solve carefully and check your work. Answer after ####.",
    "Be creative in your approach. Answer after ####.",
    "Break the problem into parts. Answer after ####.",
    "Write Python code to solve this. Print the answer.",
]

def make_seed_dags():
    return [
        # 1. Simple single-node CoT
        DAGGenome(name="dag_cot", model=CHEAP, output_node="g1", nodes=[
            DAGNode(id="g1", action="generate", system_prompt="Think step by step. Answer after ####."),
        ]),
        # 2. Fan-out → vote (3 parallel generates → merge)
        DAGGenome(name="dag_fanout_vote", model=CHEAP, output_node="vote", nodes=[
            DAGNode(id="g1", action="generate", temperature=0.0, system_prompt="Think step by step. Answer after ####."),
            DAGNode(id="g2", action="generate", temperature=0.3, system_prompt="Solve carefully. Answer after ####."),
            DAGNode(id="g3", action="generate_code", temperature=0.0),
            DAGNode(id="vote", action="vote", inputs=["g1", "g2", "g3"]),
        ]),
        # 3. Generate → verify → repair chain
        DAGGenome(name="dag_verify_repair", model=CHEAP, output_node="rep", nodes=[
            DAGNode(id="g1", action="generate", system_prompt="Solve step by step. Answer after ####."),
            DAGNode(id="ver", action="verify", inputs=["g1"]),
            DAGNode(id="rep", action="repair", inputs=["g1", "ver"]),
        ]),
        # 4. Diamond: generate → (verify + code) → merge
        DAGGenome(name="dag_diamond", model=CHEAP, output_node="merge", nodes=[
            DAGNode(id="g1", action="generate", system_prompt="Solve step by step. Answer after ####."),
            DAGNode(id="ver", action="verify", inputs=["g1"]),
            DAGNode(id="code", action="generate_code", inputs=["g1"]),
            DAGNode(id="merge", action="vote", inputs=["g1", "code"]),
        ]),
        # 5. Wide fan-out (4 generates → vote)
        DAGGenome(name="dag_wide", model=CHEAP, output_node="vote", nodes=[
            DAGNode(id="g1", action="generate", temperature=0.0, system_prompt="Think step by step. Answer after ####."),
            DAGNode(id="g2", action="generate", temperature=0.2, system_prompt="Be precise. Answer after ####."),
            DAGNode(id="g3", action="generate", temperature=0.5, system_prompt="Try a creative approach. Answer after ####."),
            DAGNode(id="g4", action="generate_code", temperature=0.0),
            DAGNode(id="vote", action="vote", inputs=["g1", "g2", "g3", "g4"]),
        ]),
    ]


# ═══════════════════════════════════════════════════════════════
# GRAPH-AWARE EVOLUTIONARY OPERATORS
# ═══════════════════════════════════════════════════════════════

def _new_id(dag):
    existing = {n.id for n in dag.nodes}
    for i in range(100):
        nid = f"n{i}"
        if nid not in existing:
            return nid
    return f"n{random.randint(100,999)}"


def mutate_dag(dag: DAGGenome) -> DAGGenome:
    new = DAGGenome(name=dag.name + "_mut", model=dag.model, output_node=dag.output_node,
                    nodes=[copy.deepcopy(n) for n in dag.nodes])
    op = random.choice(["add_node", "remove_node", "rewire", "change_action", "change_prompt"])

    if op == "add_node" and len(new.nodes) < 8:
        nid = _new_id(new)
        action = random.choice(ACTIONS)
        existing_ids = [n.id for n in new.nodes]
        # Pick 1-2 random inputs from existing nodes
        n_inputs = random.randint(0, min(2, len(existing_ids)))
        inputs = random.sample(existing_ids, n_inputs) if n_inputs > 0 else []
        node = DAGNode(id=nid, action=action, temperature=random.choice([0.0, 0.3, 0.5]),
                       system_prompt=random.choice(PROMPTS), inputs=inputs)
        new.nodes.append(node)
        # Maybe make this the output or wire it to output
        if random.random() < 0.3:
            new.output_node = nid

    elif op == "remove_node" and len(new.nodes) > 1:
        removable = [n for n in new.nodes if n.id != new.output_node]
        if removable:
            victim = random.choice(removable)
            new.nodes = [n for n in new.nodes if n.id != victim.id]
            # Remove references to victim from other nodes' inputs
            for n in new.nodes:
                n.inputs = [i for i in n.inputs if i != victim.id]

    elif op == "rewire":
        node = random.choice(new.nodes)
        other_ids = [n.id for n in new.nodes if n.id != node.id]
        if other_ids:
            if random.random() < 0.5 and node.inputs:
                # Remove a random input
                node.inputs.pop(random.randint(0, len(node.inputs) - 1))
            else:
                # Add a random input (avoid cycles by only adding from earlier topo order)
                candidate = random.choice(other_ids)
                if candidate not in node.inputs:
                    node.inputs.append(candidate)

    elif op == "change_action":
        node = random.choice(new.nodes)
        node.action = random.choice(ACTIONS)

    elif op == "change_prompt":
        node = random.choice(new.nodes)
        node.system_prompt = random.choice(PROMPTS)
        node.temperature = random.choice([0.0, 0.1, 0.3, 0.5, 0.7])

    # Ensure output_node exists
    if not any(n.id == new.output_node for n in new.nodes):
        new.output_node = new.nodes[-1].id

    return new


def crossover_dags(p1: DAGGenome, p2: DAGGenome) -> DAGGenome:
    """Crossover: take subgraph from each parent."""
    # Take first half of p1's nodes + second half of p2's nodes
    cut1 = len(p1.nodes) // 2
    cut2 = len(p2.nodes) // 2

    child_nodes = [copy.deepcopy(n) for n in p1.nodes[:max(1, cut1)]]
    p2_nodes = [copy.deepcopy(n) for n in p2.nodes[cut2:]]

    # Rename p2 nodes to avoid ID collisions
    existing_ids = {n.id for n in child_nodes}
    rename_map = {}
    for n in p2_nodes:
        if n.id in existing_ids:
            new_id = n.id + "_b"
            rename_map[n.id] = new_id
            n.id = new_id
        existing_ids.add(n.id)

    # Update input references
    for n in p2_nodes:
        n.inputs = [rename_map.get(i, i) for i in n.inputs]
        # Remove inputs that don't exist in child
        valid_ids = {nn.id for nn in child_nodes + p2_nodes}
        n.inputs = [i for i in n.inputs if i in valid_ids]

    child_nodes.extend(p2_nodes)

    output = child_nodes[-1].id if child_nodes else "g1"
    return DAGGenome(name=f"cross_{p1.name}_{p2.name}", model=p1.model,
                     output_node=output, nodes=child_nodes)


def llm_evolve_dag(dag: DAGGenome, score: float, errors: list, model: str = MID) -> DAGGenome:
    """LLM-guided DAG evolution."""
    dag_json = json.dumps(dag.to_dict(), indent=2)
    error_str = "\n".join(f"- {e.get('problem','')[:100]}: gold={e.get('gold','')}, got={e.get('predicted','')}"
                          for e in errors[:3])

    prompt = f"""Evolve this DAG agent (scored {score:.1f}%):
{dag_json}

Errors: {error_str or "None"}

DAG nodes have: id, action (generate/generate_code/verify/repair/vote/merge), temperature, system_prompt, inputs (list of node IDs).
The output_node is the final answer node. Nodes with the same depth (no dependency) run IN PARALLEL.

Improve the DAG. You can add/remove nodes, change connections, modify prompts.
Return ONLY valid JSON (same format).
"""
    result = call_llm(prompt=prompt, system="DAG architect. Return valid JSON.",
                      model=model, temperature=0.7, max_tokens=4096, json_mode=True)
    try:
        data = json.loads(result["content"])
        data["model"] = CHEAP
        return DAGGenome.from_dict(data)
    except Exception:
        return mutate_dag(dag)


# ═══════════════════════════════════════════════════════════════
# FAST EVAL (reuse from genesis)
# ═══════════════════════════════════════════════════════════════

def _eval_single_dag(dag_dict, sample, idx, benchmark):
    """Evaluate a single sample with retry on transient failures."""
    for attempt in range(3):
        result = _eval_single_dag_inner(dag_dict, sample, idx, benchmark)
        if result.get("correct") or attempt == 2:
            return result
        problem = result.get("problem", "")
        if any(s in str(problem).lower() for s in ["timeout", "connection", "rate", "error", "none"]):
            import time
            time.sleep(1 * (attempt + 1))
            continue
        return result
    return result


def _eval_single_dag_inner(dag_dict, sample, idx, benchmark):
    dag = DAGGenome.from_dict(dag_dict)
    try:
        if benchmark == "gsm8k":
            problem = sample.get("question", "")
            response = execute_dag(dag, problem)
            predicted = extract_number(response)
            gold = sample["gold_answer"]
            is_correct = predicted is not None and abs(predicted - gold) < 1e-6
            if not is_correct:
                return {"idx": idx, "correct": False, "problem": problem[:150],
                        "gold": str(gold), "predicted": str(predicted)}
            return {"idx": idx, "correct": True}
        elif benchmark == "humaneval":
            prompt = sample["prompt"]
            response = execute_dag(dag, f"Complete this Python function body:\n\n{prompt}")
            body = re.sub(r'```python\s*', '', response)
            body = re.sub(r'```\s*', '', body)
            # Strip #### math answer format if vote accidentally returned it
            body = re.sub(r'^####\s*.*$', '', body, flags=re.MULTILINE).strip()
            lines = body.split('\n')
            if lines and lines[0].strip().startswith('def '):
                i = 1
                if i < len(lines) and ('"""' in lines[i] or "'''" in lines[i]):
                    i += 1
                    while i < len(lines) and '"""' not in lines[i] and "'''" not in lines[i]:
                        i += 1
                    i += 1
                body = '\n'.join(lines[i:])
            full = sample["prompt"] + body + "\n" + sample["test"] + f"\ncheck({sample['entry_point']})"
            try:
                exec(full, {})
                return {"idx": idx, "correct": True}
            except:
                return {"idx": idx, "correct": False, "problem": prompt[:100]}
    except Exception as e:
        return {"idx": idx, "correct": False, "problem": str(e)[:100]}
    return {"idx": idx, "correct": False}


def fast_eval_dag(dag, samples, benchmark):
    dag_dict = dag.to_dict()
    details = []
    with ThreadPoolExecutor(max_workers=min(16, len(samples))) as ex:
        futures = {ex.submit(_eval_single_dag, dag_dict, s, i, benchmark): i
                   for i, s in enumerate(samples)}
        for f in as_completed(futures):
            details.append(f.result())
    correct = sum(1 for d in details if d.get("correct", False))
    errors = [d for d in details if not d.get("correct") and "problem" in d]
    score = round(correct / len(samples) * 100, 2)
    return {"score": score, "correct": correct, "total": len(samples), "errors": errors[:5]}


# ═══════════════════════════════════════════════════════════════
# EVOLUTION LOOP
# ═══════════════════════════════════════════════════════════════

def run_dag_evolve(benchmark="gsm8k", n_samples=30, pop_size=8, gens=15, elite=2, seed=42):
    from evaluate import load_gsm8k, load_humaneval
    if benchmark == "gsm8k":
        samples = load_gsm8k(n=n_samples, seed=seed)
    elif benchmark == "humaneval":
        samples = load_humaneval(n=n_samples, seed=seed)
    else:
        raise ValueError(f"Unknown: {benchmark}")

    print(f"═══ DAG-Evolve ═══")
    print(f"Benchmark: {benchmark}, Samples: {len(samples)}, Pop: {pop_size}, Gens: {gens}")

    seeds = make_seed_dags()[:pop_size]
    while len(seeds) < pop_size:
        seeds.append(mutate_dag(random.choice(seeds[:3])))

    # Eval initial pop in parallel
    population = []

    def _eval(dag):
        reset_cost_tracking()
        r = fast_eval_dag(dag, samples, benchmark)
        return dag, r, get_session_cost()

    print("── Initial ──")
    with ThreadPoolExecutor(max_workers=len(seeds)) as ex:
        for dag, r, cost in ex.map(lambda d: _eval(d), seeds):
            population.append({"dag": dag, "score": r["score"], "errors": r.get("errors", [])})
            n_nodes = len(dag.nodes)
            n_edges = sum(len(n.inputs) for n in dag.nodes)
            print(f"  {dag.name:25s} | {r['score']:5.1f}% | nodes={n_nodes} edges={n_edges}")

    best_ever = max(population, key=lambda x: x["score"])
    print(f"Best: {best_ever['dag'].name} = {best_ever['score']}%")

    for gen in range(gens):
        print(f"\n── Gen {gen+1}/{gens} ──")
        population.sort(key=lambda x: x["score"], reverse=True)
        new_pop = population[:elite]

        # Generate children in parallel
        children = []
        def _gen_child(idx):
            roll = random.random()
            if roll < 0.35:
                parent = random.choice(population[:4])
                child = llm_evolve_dag(parent["dag"], parent["score"], parent.get("errors", []))
                child.name = f"llm_g{gen+1}_{idx}"
                return child, "llm"
            elif roll < 0.55:
                parent = random.choice(population[:4])
                child = mutate_dag(parent["dag"])
                child.name = f"mut_g{gen+1}_{idx}"
                return child, "mut"
            elif roll < 0.75:
                p1, p2 = random.sample(population[:5], 2)
                child = crossover_dags(p1["dag"], p2["dag"])
                child.name = f"cross_g{gen+1}_{idx}"
                return child, "cross"
            else:
                child = mutate_dag(random.choice(make_seed_dags()))
                child.name = f"rand_g{gen+1}_{idx}"
                return child, "rand"

        with ThreadPoolExecutor(max_workers=pop_size - elite) as ex:
            gen_futures = [ex.submit(_gen_child, i) for i in range(elite, pop_size)]
            for f in as_completed(gen_futures):
                child, method = f.result()
                child.model = CHEAP
                children.append((child, method))

        # Eval children in parallel
        with ThreadPoolExecutor(max_workers=len(children)) as ex:
            eval_futures = {ex.submit(_eval, c): (c, m) for c, m in children}
            for f in as_completed(eval_futures):
                child, method = eval_futures[f]
                try:
                    dag, r, cost = f.result()
                    entry = {"dag": dag, "score": r["score"], "errors": r.get("errors", [])}
                    new_pop.append(entry)
                    marker = ""
                    if r["score"] > best_ever["score"]:
                        best_ever = entry
                        marker = " ***"
                    n_nodes = len(dag.nodes)
                    n_edges = sum(len(n.inputs) for n in dag.nodes)
                    print(f"  [{method:5s}] {dag.name:22s} | {r['score']:5.1f}% | n={n_nodes} e={n_edges}{marker}")
                except Exception as e:
                    print(f"  [{method:5s}] CRASH: {str(e)[:60]}")
                    new_pop.append({"dag": child, "score": 0, "errors": []})

        population = new_pop
        scores = [p["score"] for p in population]
        print(f"  Gen {gen+1}: best={max(scores):.1f}%, avg={sum(scores)/len(scores):.1f}%, "
              f"best_ever={best_ever['score']:.1f}%")

    print(f"\n{'═'*50}")
    print(f"DAG-EVOLVE COMPLETE: {best_ever['dag'].name} = {best_ever['score']}%")
    for n in best_ever["dag"].nodes:
        inputs = "←" + ",".join(n.inputs) if n.inputs else "(root)"
        out = " [OUTPUT]" if n.id == best_ever["dag"].output_node else ""
        print(f"  {n.id}: {n.action}(t={n.temperature}) {inputs}{out}")

    with open("best_dag.json", "w") as f:
        json.dump(best_ever["dag"].to_dict(), f, indent=2)
    return best_ever


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="gsm8k")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--pop", type=int, default=8)
    p.add_argument("--gens", type=int, default=15)
    args = p.parse_args()
    run_dag_evolve(args.benchmark, args.n, args.pop, args.gens)
