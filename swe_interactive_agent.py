"""Interactive SWE-bench agent — bash-based agentic loop.

Inspired by mini-swe-agent: give the LLM a bash shell inside the repo,
let it explore, localize, edit, and test interactively.

This is MUCH more powerful than one-shot Agentless because:
1. The agent can explore the codebase dynamically
2. It can run tests to verify its fix
3. It can iteratively refine based on test output
4. It adapts its strategy based on what it finds

The configuration (evolvable) controls:
- System prompt
- Available tools/commands
- Max turns
- Strategy hints
- Temperature per phase
"""

import os
import re
import json
import subprocess
import tempfile
import shutil
import time
from typing import Optional

from swe_llm import call_llm, call_llm_multi_turn, SONNET


# ── Default System Prompt (evolvable) ────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = """You are an expert software engineer tasked with fixing a bug in a Python repository.

You have access to a bash shell in the repository directory. You can run any command to explore the codebase, understand the bug, and create a fix.

## Available Commands
- `find_file <filename>` — Find files matching a name pattern
- `search <pattern>` — Search for a pattern in Python files (grep)
- `view_file <path> [start_line] [end_line]` — View a file with line numbers
- `str_replace <path>` — Replace exact text in a file. Format on next lines: OLD_TEXT, then separator `---`, then NEW_TEXT, then `END_REPLACE`
- `edit_file <path> <start_line> <end_line>` — Replace lines. Content follows until END_EDIT.
- `create_patch` — Generate a git diff of all changes made
- `run_tests <test_path>` — Run specific tests
- `bash <command>` — Run any bash command

## Workflow
1. First, understand the problem by reading the issue description
2. Explore the codebase to find relevant files
3. Understand the root cause
4. Make the minimal fix
5. Verify with tests if possible
6. Call `create_patch` to generate your final diff

## Rules
- Make MINIMAL changes — only fix the bug, don't refactor
- Preserve code style and conventions
- Think before you act — explain your reasoning briefly
- When you're done, call `create_patch` as your final action

Respond with a SINGLE command per turn. Format:
```
COMMAND_NAME arguments
```
For str_replace (PREFERRED for editing):
```
str_replace path/to/file.py
OLD_TEXT
exact text to find
---
NEW_TEXT
replacement text
END_REPLACE
```

For edit_file (use str_replace instead when possible):
```
edit_file path/to/file.py START END
new line 1
new line 2
END_EDIT
```"""


# ── Agent Configuration ──────────────────────────────────────────────────────

DEFAULT_INTERACTIVE_CONFIG = {
    "name": "interactive_agent",
    "model": SONNET,
    "temperature": 0.0,
    "max_tokens": 4096,
    "max_turns": 25,
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "strategy_hint": "",  # Optional hint prepended to first message
}


# ── Tool Execution ───────────────────────────────────────────────────────────

def execute_command(cmd_str: str, repo_path: str, timeout: int = 30) -> str:
    """Execute a tool command in the repo context. Returns output string."""
    cmd_str = cmd_str.strip()
    if not cmd_str:
        return "[empty command]"

    # Parse command
    parts = cmd_str.split(None, 1)
    cmd_name = parts[0] if parts else ""
    cmd_args = parts[1] if len(parts) > 1 else ""

    try:
        if cmd_name == "find_file":
            pattern = cmd_args.strip() or "*.py"
            result = subprocess.run(
                ["find", ".", "-type", "f", "-name", pattern, "-not", "-path", r"*/\.*"],
                cwd=repo_path, capture_output=True, text=True, timeout=timeout
            )
            output = result.stdout.strip()
            lines = output.split("\n")
            if len(lines) > 50:
                return "\n".join(lines[:50]) + f"\n... ({len(lines)} total matches)"
            return output or "[no matches]"

        elif cmd_name == "search":
            pattern = cmd_args.strip()
            if not pattern:
                return "[error: search requires a pattern]"
            result = subprocess.run(
                ["grep", "-rn", "--include=*.py", pattern, "."],
                cwd=repo_path, capture_output=True, text=True, timeout=timeout
            )
            output = result.stdout.strip()
            lines = output.split("\n")
            if len(lines) > 40:
                return "\n".join(lines[:40]) + f"\n... ({len(lines)} total matches)"
            return output or "[no matches]"

        elif cmd_name == "view_file":
            args_parts = cmd_args.strip().split()
            if not args_parts:
                return "[error: view_file requires a file path]"
            filepath = args_parts[0]
            start = int(args_parts[1]) if len(args_parts) > 1 else 1
            end = int(args_parts[2]) if len(args_parts) > 2 else start + 100

            full_path = os.path.join(repo_path, filepath)
            if not os.path.exists(full_path):
                return f"[file not found: {filepath}]"

            with open(full_path, "r", errors="replace") as f:
                all_lines = f.readlines()

            selected = all_lines[max(0, start-1):end]
            numbered = ""
            for i, line in enumerate(selected, start=max(1, start)):
                numbered += f"{i:4d} | {line}"

            return f"[{filepath}] ({len(all_lines)} lines total)\n{numbered}"

        elif cmd_name == "str_replace":
            return _handle_str_replace(cmd_args, cmd_str, repo_path)

        elif cmd_name == "edit_file":
            return _handle_edit(cmd_args, cmd_str, repo_path)

        elif cmd_name == "create_patch":
            result = subprocess.run(
                ["git", "diff"],
                cwd=repo_path, capture_output=True, text=True, timeout=timeout
            )
            diff = result.stdout.strip()
            if not diff:
                return "[no changes detected — did you edit any files?]"
            return f"[PATCH GENERATED]\n{diff}"

        elif cmd_name == "run_tests":
            test_path = cmd_args.strip()
            result = subprocess.run(
                ["python", "-m", "pytest", test_path, "-x", "--tb=short", "-q"],
                cwd=repo_path, capture_output=True, text=True, timeout=120
            )
            output = result.stdout + result.stderr
            # Truncate long test output
            if len(output) > 3000:
                output = output[:1500] + "\n...[truncated]...\n" + output[-1500:]
            return output or "[tests produced no output]"

        elif cmd_name == "bash":
            bash_cmd = cmd_args.strip()
            if not bash_cmd:
                return "[error: bash requires a command]"
            # Safety: block dangerous commands
            dangerous = ["rm -rf /", "rm -rf ~", ":(){ :|:& };:", "mkfs", "dd if="]
            if any(d in bash_cmd for d in dangerous):
                return "[blocked: dangerous command]"
            result = subprocess.run(
                bash_cmd, shell=True,
                cwd=repo_path, capture_output=True, text=True, timeout=timeout
            )
            output = (result.stdout + result.stderr).strip()
            if len(output) > 3000:
                output = output[:1500] + "\n...[truncated]...\n" + output[-1500:]
            return output or "[command produced no output]"

        else:
            return f"[unknown command: {cmd_name}]. Available: find_file, search, view_file, edit_file, create_patch, run_tests, bash"

    except subprocess.TimeoutExpired:
        return f"[command timed out after {timeout}s]"
    except Exception as e:
        return f"[error: {str(e)}]"


def _handle_str_replace(cmd_args: str, full_cmd: str, repo_path: str) -> str:
    """Handle str_replace — find exact text and replace it. More robust than line editing."""
    lines = full_cmd.split("\n")
    filepath = lines[0].split(None, 1)[1].strip() if len(lines[0].split(None, 1)) > 1 else ""

    if not filepath:
        return "[error: str_replace requires a file path]"

    full_path = os.path.join(repo_path, filepath)
    if not os.path.exists(full_path):
        return f"[file not found: {filepath}]"

    # Parse OLD_TEXT and NEW_TEXT sections
    old_text = []
    new_text = []
    section = None
    for line in lines[1:]:
        stripped = line.strip()
        if stripped == "OLD_TEXT":
            section = "old"
            continue
        elif stripped == "---":
            section = "transition"
            continue
        elif stripped == "NEW_TEXT":
            section = "new"
            continue
        elif stripped == "END_REPLACE":
            break

        if section == "old" or (section is None and not old_text):
            old_text.append(line)
        elif section in ("new", "transition"):
            new_text.append(line)
            section = "new"

    old_str = "\n".join(old_text)
    new_str = "\n".join(new_text)

    if not old_str:
        return "[error: OLD_TEXT section is empty]"

    with open(full_path, "r", errors="replace") as f:
        content = f.read()

    if old_str not in content:
        # Try with stripped whitespace matching
        stripped_old = old_str.strip()
        if stripped_old in content:
            content = content.replace(stripped_old, new_str.strip(), 1)
        else:
            return f"[error: OLD_TEXT not found in {filepath}. Make sure you copy the exact text including whitespace.]"
    else:
        content = content.replace(old_str, new_str, 1)

    with open(full_path, "w") as f:
        f.write(content)

    return f"[str_replace: replaced text in {filepath} ({len(old_str)} chars → {len(new_str)} chars)]"


def _handle_edit(cmd_args: str, full_cmd: str, repo_path: str) -> str:
    """Handle edit_file command — replaces lines in a file."""
    # Parse: edit_file path start end\nnew_content\nEND_EDIT
    lines = full_cmd.split("\n")
    header = lines[0]  # edit_file path start end
    header_parts = header.split()

    if len(header_parts) < 4:
        return "[error: edit_file requires: path start_line end_line]"

    filepath = header_parts[1]
    try:
        start_line = int(header_parts[2])
        end_line = int(header_parts[3])
    except ValueError:
        return "[error: start_line and end_line must be integers]"

    # Find content between header and END_EDIT
    content_lines = []
    for i, line in enumerate(lines[1:], 1):
        if line.strip() == "END_EDIT":
            break
        content_lines.append(line)

    full_path = os.path.join(repo_path, filepath)
    if not os.path.exists(full_path):
        return f"[file not found: {filepath}]"

    with open(full_path, "r", errors="replace") as f:
        file_lines = f.readlines()

    if start_line < 1 or end_line > len(file_lines):
        return f"[error: line range {start_line}-{end_line} out of bounds (file has {len(file_lines)} lines)]"

    # Replace lines: keep everything before start, insert new content, keep everything after end
    before = file_lines[:start_line - 1]
    after = file_lines[end_line:]

    # Each new line needs a newline at the end
    new_lines = []
    for cl in content_lines:
        if not cl.endswith("\n"):
            cl += "\n"
        new_lines.append(cl)

    new_file = before + new_lines + after

    with open(full_path, "w") as f:
        f.writelines(new_file)

    return f"[edited {filepath}: replaced lines {start_line}-{end_line} ({end_line - start_line + 1} lines → {len(content_lines)} lines)]"


# ── Interactive Agent Loop ───────────────────────────────────────────────────

def parse_command_from_response(response: str) -> str:
    """Extract the command from agent's response."""
    # Look for command in code block
    code_match = re.search(r'```\w*\s*\n(.*?)```', response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()

    # Look for known command names at start of lines
    known_cmds = ["find_file", "search", "view_file", "str_replace", "edit_file",
                  "create_patch", "run_tests", "bash"]
    for line in response.split("\n"):
        stripped = line.strip()
        for cmd in known_cmds:
            if stripped.startswith(cmd):
                # For edit_file, capture everything until END_EDIT
                if cmd == "edit_file":
                    idx = response.find(stripped)
                    end_idx = response.find("END_EDIT", idx)
                    if end_idx != -1:
                        return response[idx:end_idx + len("END_EDIT")]
                # For str_replace, capture everything until END_REPLACE
                if cmd == "str_replace":
                    idx = response.find(stripped)
                    end_idx = response.find("END_REPLACE", idx)
                    if end_idx != -1:
                        return response[idx:end_idx + len("END_REPLACE")]
                return stripped

    return ""


def solve_interactive(instance: dict, config: dict = None,
                      work_dir: str = None) -> dict:
    """Solve a SWE-bench instance using interactive agent loop.

    The agent gets a bash-like shell and can explore/edit/test iteratively.
    """
    if config is None:
        config = DEFAULT_INTERACTIVE_CONFIG.copy()

    instance_id = instance["instance_id"]
    repo = instance["repo"]
    base_commit = instance["base_commit"]
    problem_statement = instance["problem_statement"]
    hints = instance.get("hints_text", "")

    max_turns = config.get("max_turns", 25)
    system_prompt = config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    model = config.get("model", SONNET)
    temperature = config.get("temperature", 0.0)

    # Clone repo
    cleanup = False
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="swe_interactive_")
        cleanup = True

    try:
        from swe_agent import clone_repo
        repo_path = clone_repo(repo, base_commit, work_dir)

        # Build initial message
        first_msg = f"""Fix the following issue in the {repo} repository:

{problem_statement}
"""
        if hints:
            first_msg += f"\nHints: {hints}\n"

        strategy = config.get("strategy_hint", "")
        if strategy:
            first_msg += f"\nSuggested approach: {strategy}\n"

        first_msg += "\nStart by exploring the repository to understand the codebase and locate the relevant files."

        # Conversation history
        messages = [{"role": "user", "content": first_msg}]

        final_patch = ""
        trajectory = []  # (turn, command, output)
        total_cost = 0.0

        for turn in range(max_turns):
            # Get agent response
            resp = call_llm_multi_turn(
                messages=messages,
                system=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=config.get("max_tokens", 4096),
            )

            agent_response = resp["content"]
            total_cost += resp["cost_usd"]

            # Extract command
            command = parse_command_from_response(agent_response)

            if not command:
                # Agent didn't give a command — prompt it
                messages.append({"role": "assistant", "content": agent_response})
                messages.append({"role": "user", "content": "Please provide a command to execute. Available commands: find_file, search, view_file, edit_file, create_patch, run_tests, bash"})
                continue

            # Execute command
            output = execute_command(command, repo_path)
            trajectory.append({"turn": turn, "command": command[:200], "output": output[:500]})

            # Check if patch was generated
            if command.startswith("create_patch") and "[PATCH GENERATED]" in output:
                final_patch = output.replace("[PATCH GENERATED]\n", "")
                break

            # Add to conversation
            messages.append({"role": "assistant", "content": agent_response})

            # Truncate output for context window management
            if len(output) > 4000:
                output = output[:2000] + "\n...[truncated]...\n" + output[-2000:]

            # Smart wrap-up: nudge agent to finish if it has edits and is running low
            has_edits = any("edit" in t.get("command", "").lower() or
                           "str_replace" in t.get("command", "").lower()
                           for t in trajectory)
            turns_left = max_turns - turn - 1

            if has_edits and turns_left <= 3:
                suffix = "\n\n⚠️ You have made edits and only have {} turns left. Run `create_patch` NOW to save your work.".format(turns_left)
            elif has_edits and turns_left <= 7:
                suffix = "\n\nYou've made edits. When ready, run `create_patch` to generate the final diff."
            else:
                suffix = "\n\nContinue. When you're done with your fix, run `create_patch` to generate the final diff."

            messages.append({"role": "user", "content": f"Command output:\n```\n{output}\n```{suffix}"})

            # Context window management: if messages get too long, summarize early turns
            total_chars = sum(len(m["content"]) for m in messages)
            if total_chars > 100000 and len(messages) > 10:
                # Keep first 2 and last 8 messages
                messages = messages[:2] + [
                    {"role": "user", "content": "[Earlier conversation turns omitted for context management]"}
                ] + messages[-8:]

        # If agent didn't call create_patch, get diff anyway
        if not final_patch:
            result = subprocess.run(
                ["git", "diff"], cwd=repo_path,
                capture_output=True, text=True, timeout=30
            )
            final_patch = result.stdout.strip()

        return {
            "instance_id": instance_id,
            "model_name_or_path": config.get("name", "interactive_agent"),
            "model_patch": final_patch,
            "trajectory": trajectory,
            "turns_used": len(trajectory),
            "total_cost": total_cost,
        }

    except Exception as e:
        return {
            "instance_id": instance_id,
            "model_name_or_path": config.get("name", "interactive_agent"),
            "model_patch": "",
            "error": str(e),
        }
    finally:
        if cleanup and work_dir and os.path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
