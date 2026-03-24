"""SWE-bench agent — generates patches for GitHub issues.

Implements an Agentless-inspired pipeline:
1. Localize: Find relevant files/functions from the problem statement
2. Repair: Generate a patch to fix the issue
3. Validate: Optionally self-check the patch

This is the BASE agent. The evolutionary system evolves configurations
that modify how this agent operates (prompts, strategy, tools, topology).
"""

import os
import re
import json
import subprocess
import tempfile
import shutil
from typing import Optional
from swe_llm import call_llm, call_llm_multi_turn, AGENT_MODEL


# ── Agent Configuration (evolvable) ─────────────────────────────────────────

DEFAULT_CONFIG = {
    "name": "baseline_agentless",
    "model": AGENT_MODEL,
    "temperature": 0.0,
    "max_tokens": 8192,

    # Localization config
    "localize_strategy": "file_then_function",  # file_then_function | grep_based | ast_based
    "localize_prompt": """You are a senior software engineer debugging a GitHub issue.

Given the problem statement and repository structure, identify the most likely files and functions that need to be modified.

Think step by step:
1. What is the bug or feature request about?
2. Which modules/packages are relevant?
3. Which specific files likely contain the buggy code?
4. Which functions/methods need changes?

Return your analysis as JSON:
{
    "analysis": "brief explanation of the bug",
    "files": ["path/to/file1.py", "path/to/file2.py"],
    "functions": ["ClassName.method_name", "function_name"],
    "confidence": 0.0-1.0
}""",

    # Repair config
    "repair_strategy": "direct",  # direct | cot_then_patch | multi_candidate
    "repair_prompt": """You are a senior software engineer fixing a GitHub issue.

Given the problem statement and the relevant source code, generate a minimal patch that fixes the issue.

Rules:
- Make the MINIMUM changes needed to fix the bug
- Do NOT add unnecessary imports, comments, or refactoring
- Preserve existing code style and conventions
- Think about edge cases the fix needs to handle
- Output ONLY a unified diff (git diff format)

The diff should be applicable with `git apply`. Use the exact file paths shown.
Format:
```diff
diff --git a/path/to/file.py b/path/to/file.py
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -line,count +line,count @@
 context line
-removed line
+added line
 context line
```""",

    # Validation config
    "validate": True,
    "validate_prompt": """Review this patch for correctness:

1. Does it address the issue described?
2. Are there any syntax errors?
3. Could it break existing functionality?
4. Are edge cases handled?

If the patch looks correct, respond with: APPROVE
If it has issues, respond with: REJECT: <reason>""",

    # Multi-candidate
    "num_candidates": 1,
    "temperatures": [0.0],
}


# Global repo cache directory — avoids re-cloning the same repo
_REPO_CACHE_DIR = os.path.join(tempfile.gettempdir(), "swe_repo_cache")


def clone_repo(repo: str, base_commit: str, work_dir: str) -> str:
    """Clone repo at specific commit into work_dir. Uses cache when possible."""
    repo_path = os.path.join(work_dir, repo.replace("/", "__"))
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)

    url = f"https://github.com/{repo}.git"
    cache_path = os.path.join(_REPO_CACHE_DIR, repo.replace("/", "__"))

    # Try to use cached repo
    if os.path.exists(cache_path):
        # Copy from cache and checkout the right commit
        shutil.copytree(cache_path, repo_path, symlinks=True)
        # Reset any changes from previous use
        subprocess.run(["git", "checkout", "--", "."], cwd=repo_path,
                       capture_output=True, timeout=30)
        subprocess.run(["git", "clean", "-fd"], cwd=repo_path,
                       capture_output=True, timeout=30)
        # Fetch the specific commit
        subprocess.run(
            ["git", "fetch", "--depth", "1", "origin", base_commit],
            cwd=repo_path, capture_output=True, text=True, timeout=120
        )
        checkout = subprocess.run(
            ["git", "checkout", base_commit],
            cwd=repo_path, capture_output=True, text=True, timeout=30
        )
        if checkout.returncode == 0:
            return repo_path
        # If checkout failed, try unshallow
        subprocess.run(["git", "fetch", "--unshallow"],
                       cwd=repo_path, capture_output=True, timeout=300)
        checkout = subprocess.run(
            ["git", "checkout", base_commit],
            cwd=repo_path, capture_output=True, text=True, timeout=30
        )
        if checkout.returncode == 0:
            return repo_path
        # Cache is stale, remove and re-clone
        shutil.rmtree(repo_path, ignore_errors=True)

    # Fresh clone
    result = subprocess.run(
        ["git", "clone", "--depth", "1", url, repo_path],
        capture_output=True, text=True, timeout=180
    )
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr[:200]}")

    # Fetch the specific commit
    subprocess.run(
        ["git", "fetch", "--depth", "1", "origin", base_commit],
        cwd=repo_path, capture_output=True, text=True, timeout=120
    )
    checkout = subprocess.run(
        ["git", "checkout", base_commit],
        cwd=repo_path, capture_output=True, text=True, timeout=30
    )
    if checkout.returncode != 0:
        subprocess.run(["git", "fetch", "--unshallow"],
                       cwd=repo_path, capture_output=True, timeout=300)
        subprocess.run(["git", "checkout", base_commit],
                       cwd=repo_path, capture_output=True, timeout=30)

    # Save to cache (first time only)
    os.makedirs(_REPO_CACHE_DIR, exist_ok=True)
    if not os.path.exists(cache_path):
        try:
            shutil.copytree(repo_path, cache_path, symlinks=True)
        except Exception:
            pass  # Non-critical — caching is best-effort

    return repo_path


def get_repo_structure(repo_path: str, max_depth: int = 3) -> str:
    """Get directory tree of the repo."""
    result = subprocess.run(
        ["find", ".", "-type", "f", "-name", "*.py", "-not", "-path", r"*/\.*",
         "-not", "-path", "*/test*", "-not", "-path", "*/__pycache__/*"],
        cwd=repo_path, capture_output=True, text=True, timeout=30
    )
    files = sorted(result.stdout.strip().split("\n")[:200])
    return "\n".join(files)


def get_file_content(repo_path: str, filepath: str, max_lines: int = 500) -> str:
    """Read a file from the repo with line numbers."""
    full_path = os.path.join(repo_path, filepath)
    if not os.path.exists(full_path):
        return f"[FILE NOT FOUND: {filepath}]"
    with open(full_path, "r", errors="replace") as f:
        lines = f.readlines()[:max_lines]
    return "".join(f"{i+1:4d} | {line}" for i, line in enumerate(lines))


def grep_repo(repo_path: str, pattern: str, max_results: int = 30) -> str:
    """Search for pattern in repo."""
    try:
        result = subprocess.run(
            ["grep", "-rn", "--include=*.py", "-l", pattern, "."],
            cwd=repo_path, capture_output=True, text=True, timeout=15
        )
        files = result.stdout.strip().split("\n")[:max_results]
        return "\n".join(files) if files[0] else "[no matches]"
    except Exception:
        return "[grep failed]"


def localize(problem_statement: str, repo_path: str, config: dict) -> dict:
    """Phase 1: Identify which files/functions to modify."""
    structure = get_repo_structure(repo_path)

    prompt = f"""Repository structure (Python files):
{structure}

Problem statement:
{problem_statement}

{config.get('localize_prompt', DEFAULT_CONFIG['localize_prompt'])}"""

    resp = call_llm(
        prompt=prompt,
        model=config.get("model", AGENT_MODEL),
        temperature=config.get("temperature", 0.0),
        max_tokens=config.get("max_tokens", 4096),
    )

    # Parse JSON from response
    content = resp["content"]
    try:
        # Try to extract JSON from response
        json_match = re.search(r'\{[^{}]*"files"[^{}]*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            # Fallback: extract file paths
            files = re.findall(r'[\w/]+\.py', content)
            result = {"files": files[:5], "analysis": content[:200], "confidence": 0.5}
    except json.JSONDecodeError:
        files = re.findall(r'[\w/]+\.py', content)
        result = {"files": files[:5], "analysis": content[:200], "confidence": 0.5}

    result["_raw"] = content
    result["_cost"] = resp["cost_usd"]
    return result


def repair(problem_statement: str, repo_path: str, localization: dict, config: dict) -> str:
    """Phase 2: Generate a patch."""
    # Gather source code for identified files
    source_sections = []
    for filepath in localization.get("files", [])[:5]:
        # Clean the path
        fp = filepath.lstrip("./")
        content = get_file_content(repo_path, fp)
        if "[FILE NOT FOUND" not in content:
            source_sections.append(f"=== {fp} ===\n{content}")

    if not source_sections:
        # Fallback: grep for keywords from problem statement
        keywords = re.findall(r'\b[A-Z][a-z]+[A-Z]\w+\b|\b[a-z_]+\b', problem_statement[:500])
        for kw in keywords[:5]:
            matches = grep_repo(repo_path, kw)
            if matches != "[no matches]":
                for match_file in matches.split("\n")[:2]:
                    fp = match_file.lstrip("./")
                    content = get_file_content(repo_path, fp, max_lines=300)
                    if "[FILE NOT FOUND" not in content:
                        source_sections.append(f"=== {fp} ===\n{content}")
                break

    source_code = "\n\n".join(source_sections[:3])  # Limit to 3 files

    prompt = f"""Problem statement:
{problem_statement}

Analysis: {localization.get('analysis', 'N/A')}

Relevant source code:
{source_code}

{config.get('repair_prompt', DEFAULT_CONFIG['repair_prompt'])}"""

    resp = call_llm(
        prompt=prompt,
        model=config.get("model", AGENT_MODEL),
        temperature=config.get("temperature", 0.0),
        max_tokens=config.get("max_tokens", 8192),
    )

    return resp["content"]


def extract_diff(response: str) -> str:
    """Extract unified diff from LLM response."""
    # Try to find diff block in code fence
    diff_match = re.search(r'```(?:diff)?\s*\n((?:diff --git|---|\+\+\+|@@).*?)```', response, re.DOTALL)
    if diff_match:
        return diff_match.group(1).strip()

    # Try any code fence that contains diff-like content
    code_blocks = re.findall(r'```\w*\s*\n(.*?)```', response, re.DOTALL)
    for block in code_blocks:
        if any(marker in block for marker in ["diff --git", "--- a/", "+++ b/", "@@ "]):
            return block.strip()

    # Try to find raw diff lines outside code fences
    lines = response.split("\n")
    diff_lines = []
    in_diff = False
    for line in lines:
        if line.startswith("diff --git") or (line.startswith("--- a/") and not in_diff):
            in_diff = True
        if in_diff:
            diff_lines.append(line)

    if diff_lines:
        return "\n".join(diff_lines)

    return ""  # Return empty — caller should retry


def retry_for_diff(problem_statement: str, source_code: str,
                   first_response: str, config: dict) -> str:
    """If the first attempt didn't produce a diff, explicitly ask for one."""
    prompt = f"""Your previous response did not contain a valid unified diff (git diff format).

Previous analysis:
{first_response[:2000]}

Now, please output ONLY the unified diff patch. No explanation, no analysis.
The diff must start with "diff --git" and use the exact file paths from the source code.

Problem: {problem_statement[:500]}

Source files available:
{source_code[:3000]}

Output ONLY the diff:
```diff
diff --git a/path/to/file.py b/path/to/file.py
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -line,count +line,count @@
 context
-old
+new
 context
```"""

    resp = call_llm(
        prompt=prompt,
        model=config.get("model", AGENT_MODEL),
        temperature=0.0,
        max_tokens=config.get("max_tokens", 8192),
    )
    return resp["content"]


def validate_patch(problem_statement: str, patch: str, config: dict) -> tuple:
    """Phase 3: Validate the patch. Returns (approved: bool, reason: str)."""
    if not config.get("validate", True):
        return True, "validation disabled"

    prompt = f"""Problem statement:
{problem_statement}

Generated patch:
```diff
{patch}
```

{config.get('validate_prompt', DEFAULT_CONFIG['validate_prompt'])}"""

    resp = call_llm(
        prompt=prompt,
        model=config.get("model", AGENT_MODEL),
        temperature=0.0,
        max_tokens=2048,
    )

    content = resp["content"]
    if "APPROVE" in content.upper():
        return True, "approved"
    else:
        reason = content.split("REJECT:")[-1].strip() if "REJECT" in content.upper() else content[:200]
        return False, reason


def solve_instance(instance: dict, config: dict = None, work_dir: str = None) -> dict:
    """Solve a single SWE-bench instance. Returns dict with model_patch and metadata."""
    if config is None:
        config = DEFAULT_CONFIG.copy()

    instance_id = instance["instance_id"]
    repo = instance["repo"]
    base_commit = instance["base_commit"]
    problem_statement = instance["problem_statement"]
    hints = instance.get("hints_text", "")

    if hints:
        problem_statement += f"\n\nHints: {hints}"

    # Use temp dir if no work_dir provided
    cleanup = False
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="swe_")
        cleanup = True

    try:
        # Clone repo
        repo_path = clone_repo(repo, base_commit, work_dir)

        # Phase 1: Localize
        localization = localize(problem_statement, repo_path, config)

        # Phase 2: Repair (possibly multiple candidates)
        num_candidates = config.get("num_candidates", 1)
        temps = config.get("temperatures", [0.0])

        best_patch = None
        patches = []

        # Gather source code for retry_for_diff
        source_sections = []
        for fp in localization.get("files", [])[:3]:
            fp = fp.lstrip("./")
            content = get_file_content(repo_path, fp, max_lines=200)
            if "[FILE NOT FOUND" not in content:
                source_sections.append(f"=== {fp} ===\n{content}")
        source_code_str = "\n\n".join(source_sections)

        for i in range(num_candidates):
            temp_config = config.copy()
            temp_config["temperature"] = temps[i % len(temps)]

            response = repair(problem_statement, repo_path, localization, temp_config)
            patch = extract_diff(response)

            # Retry if no valid diff produced
            if not patch:
                retry_resp = retry_for_diff(
                    problem_statement, source_code_str, response, temp_config
                )
                patch = extract_diff(retry_resp)

            patches.append(patch)

            # Phase 3: Validate
            if config.get("validate", True) and num_candidates > 1 and patch:
                approved, reason = validate_patch(problem_statement, patch, config)
                if approved:
                    best_patch = patch
                    break

        if best_patch is None:
            # Pick first non-empty patch
            for p in patches:
                if p:
                    best_patch = p
                    break
            if best_patch is None:
                best_patch = patches[0] if patches else ""

        return {
            "instance_id": instance_id,
            "model_name_or_path": config.get("name", "baseline"),
            "model_patch": best_patch,
            "localization": localization,
            "num_candidates": len(patches),
        }

    except Exception as e:
        return {
            "instance_id": instance_id,
            "model_name_or_path": config.get("name", "baseline"),
            "model_patch": "",
            "error": str(e),
        }
    finally:
        if cleanup and work_dir and os.path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)


def solve_batch(instances: list, config: dict = None,
                max_workers: int = 4, work_dir: str = None) -> list:
    """Solve multiple SWE-bench instances."""
    import concurrent.futures

    results = []

    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="swe_batch_")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for inst in instances:
            inst_dir = os.path.join(work_dir, inst["instance_id"].replace("/", "__"))
            os.makedirs(inst_dir, exist_ok=True)
            future = executor.submit(solve_instance, inst, config, inst_dir)
            futures[future] = inst["instance_id"]

        for future in concurrent.futures.as_completed(futures):
            iid = futures[future]
            try:
                result = future.result(timeout=600)
                results.append(result)
                print(f"  ✓ {iid}: patch generated ({len(result.get('model_patch', ''))} chars)")
            except Exception as e:
                results.append({
                    "instance_id": iid,
                    "model_name_or_path": "baseline",
                    "model_patch": "",
                    "error": str(e),
                })
                print(f"  ✗ {iid}: {e}")

    return results
