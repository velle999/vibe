import os
import subprocess
import glob as glob_module
import re
from pathlib import Path


# ── Tool definitions (OpenAI function-calling schema) ──────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full contents of a file. Always read a file before editing it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write (overwrite) a file with new content. Use for creating new files or full rewrites.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to write"},
                    "content": {"type": "string", "description": "Full file content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact string in a file. old_string must be unique in the file. Prefer this over write_file for targeted edits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "old_string": {"type": "string", "description": "Exact text to find and replace"},
                    "new_string": {"type": "string", "description": "Replacement text"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command and return stdout+stderr. Use for tests, builds, installs, git, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py'"},
                    "path": {"type": "string", "description": "Root directory to search (default: cwd)"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents for a regex pattern. Returns matching lines with file and line number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "Directory or file to search (default: cwd)"},
                    "file_glob": {"type": "string", "description": "Only search files matching this glob, e.g. '*.py'"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories at a path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory to list (default: cwd)"},
                },
                "required": [],
            },
        },
    },
]


# ── Tool implementations ────────────────────────────────────────────────────────

def read_file(path: str) -> str:
    p = Path(path).expanduser()
    if not p.exists():
        return f"Error: file not found: {path}"
    if not p.is_file():
        return f"Error: not a file: {path}"
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        numbered = "\n".join(f"{i+1:4d}  {line}" for i, line in enumerate(lines))
        return f"<file path={path} lines={len(lines)}>\n{numbered}\n</file>"
    except Exception as e:
        return f"Error reading {path}: {e}"


def write_file(path: str, content: str) -> str:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        p.write_text(content, encoding="utf-8")
        lines = content.count("\n") + 1
        _edit_success_counts.pop(str(p.resolve()), None)
        return f"Written {lines} lines to {path}"
    except Exception as e:
        return f"Error writing {path}: {e}"


_edit_fail_counts: dict[str, int] = {}
_edit_success_counts: dict[str, int] = {}
_EDIT_CHURN_THRESHOLD = 5  # warn after this many successive edits to the same file


def edit_file(path: str, old_string: str, new_string: str) -> str:
    p = Path(path).expanduser()
    if not p.exists():
        return f"Error: file not found: {path}"
    try:
        original = p.read_text(encoding="utf-8", errors="replace")
        count = original.count(old_string)
        if count == 0:
            key = str(p.resolve())
            _edit_fail_counts[key] = _edit_fail_counts.get(key, 0) + 1
            fails = _edit_fail_counts[key]
            # Show the full file so the model has everything it needs for write_file
            preview = original.replace("\t", "→")
            header = (
                f"STOP. edit_file has failed {fails} time(s) on {path}. "
                f"old_string does not exist in the file. "
                f"YOU MUST use write_file NOW — do not call edit_file again on this file. "
                f"Here is the complete current file content to use as the base for write_file:\n\n"
            )
            return header + preview
        if count > 1:
            return f"Error: old_string matches {count} times in {path} — make it more specific"
        updated = original.replace(old_string, new_string, 1)
        p.write_text(updated, encoding="utf-8")
        key = str(p.resolve())
        _edit_fail_counts.pop(key, None)
        _edit_success_counts[key] = _edit_success_counts.get(key, 0) + 1
        n = _edit_success_counts[key]
        msg = f"Replaced 1 occurrence in {path}"
        if n >= _EDIT_CHURN_THRESHOLD:
            msg += (
                f"\n\nNote: you have now made {n} successive edits to this file. "
                f"If you are still building or fixing it, use write_file to rewrite the "
                f"whole file at once — it is faster and less error-prone than many small edits."
            )
        return msg
    except Exception as e:
        return f"Error editing {path}: {e}"


_BASH_MAX_LINES = 100
_tty_blocked: set[str] = {}  # script paths that have already timed out this session

_TTY_BLOCK_ERROR = (
    "BLOCKED: This script has already timed out once this session, which means it requires "
    "an interactive terminal (TTY). Running it again will not work. "
    "Do NOT call bash on this script again. "
    "Tell the user the script is complete and they should run it directly in their terminal."
)


def _extract_script_path(command: str) -> str | None:
    """Return the script path if the command is executing a shell script file."""
    import re
    # Match: ./foo.sh, bash foo.sh, sh ./foo.sh, python foo.py, etc.
    m = re.search(r'(?:^|(?:bash|sh|python3?|node|ruby|perl)\s+)(\.?\.?/?\S+\.(?:sh|py|js|rb|pl))', command)
    return m.group(1) if m else None


def bash(command: str, timeout: int = 30) -> str:
    script = _extract_script_path(command)
    if script and script in _tty_blocked:
        return _TTY_BLOCK_ERROR

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += result.stderr
        if result.returncode != 0:
            output += f"\n[exit code {result.returncode}]"
        output = output.strip() or "[no output]"

        # Truncate to avoid bloating the context with noisy output
        lines = output.splitlines()
        if len(lines) > _BASH_MAX_LINES:
            half = _BASH_MAX_LINES // 2
            omitted = len(lines) - _BASH_MAX_LINES
            output = (
                "\n".join(lines[:half])
                + f"\n\n... ({omitted} lines omitted) ...\n\n"
                + "\n".join(lines[-half:])
            )

        # Detect common no-TTY failures for interactive programs
        if any(phrase in output for phrase in (
            "cbreak() returned ERR", "nocbreak() returned ERR",
            "setupterm: could not find terminal", "not a terminal",
        )):
            return (
                "Error: this program requires an interactive terminal (TTY) and cannot be "
                "run through the bash tool. Write the code and instruct the user to run it "
                "directly in their terminal instead."
            )

        return output
    except subprocess.TimeoutExpired:
        if script:
            _tty_blocked.add(script)
        return (
            f"Error: command timed out after {timeout}s. "
            "This means the program requires an interactive terminal (TTY) — "
            "it is waiting for input or running an event/game loop that never exits. "
            "This script is now BLOCKED from running via bash for the rest of this session. "
            "Do NOT retry. Do NOT modify the script to add timeouts. "
            "Tell the user the script is ready and they should run it directly in their terminal."
        )
    except Exception as e:
        return f"Error: {e}"


def glob(pattern: str, path: str | None = None) -> str:
    root = Path(path).expanduser() if path else Path.cwd()
    try:
        matches = sorted(root.glob(pattern))
        if not matches:
            return "No files matched."
        return "\n".join(str(m.relative_to(root)) for m in matches[:200])
    except Exception as e:
        return f"Error: {e}"


def grep(pattern: str, path: str | None = None, file_glob: str | None = None) -> str:
    root = Path(path).expanduser() if path else Path.cwd()
    try:
        compiled = re.compile(pattern)
    except re.error as e:
        return f"Invalid regex: {e}"

    results = []
    file_iter = root.rglob(file_glob) if file_glob else root.rglob("*")

    for fp in file_iter:
        if not fp.is_file():
            continue
        try:
            for i, line in enumerate(fp.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if compiled.search(line):
                    rel = fp.relative_to(root)
                    results.append(f"{rel}:{i}: {line.rstrip()}")
        except Exception:
            continue
        if len(results) >= 500:
            results.append("... (truncated at 500 results)")
            break

    return "\n".join(results) if results else "No matches found."


def list_dir(path: str | None = None) -> str:
    p = Path(path).expanduser() if path else Path.cwd()
    if not p.exists():
        return f"Error: path not found: {path}"
    try:
        entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
        lines = []
        for e in entries:
            if e.is_dir():
                lines.append(f"  {e.name}/")
            else:
                size = e.stat().st_size
                lines.append(f"  {e.name}  ({size:,} bytes)")
        return "\n".join(lines) or "(empty directory)"
    except Exception as e:
        return f"Error: {e}"


# ── Dispatch ───────────────────────────────────────────────────────────────────

TOOL_MAP = {
    "read_file": lambda args: read_file(**args),
    "write_file": lambda args: write_file(**args),
    "edit_file": lambda args: edit_file(**args),
    "bash": lambda args: bash(**args),
    "glob": lambda args: glob(**args),
    "grep": lambda args: grep(**args),
    "list_dir": lambda args: list_dir(**args),
}


def execute_tool(name: str, args: dict) -> str:
    fn = TOOL_MAP.get(name)
    if fn is None:
        return f"Error: unknown tool '{name}'"
    try:
        return fn(args)
    except TypeError as e:
        return f"Error: bad arguments for {name}: {e}"
