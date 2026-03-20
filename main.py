"""
Vibe Code — local AI coding assistant
REPL entry point with slash commands
"""

import os
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich import box

import vibe.config as cfg
from vibe.llm import VibeModel
from vibe.ui import (
    get_input,
    print_welcome,
    print_help,
    stream_response,
    print_error,
    print_info,
    console,
)
from vibe.system import (
    sys_info,
    gpu_info,
    net_info,
    ps_list,
    kill_process,
    service_control,
    services_list,
    open_file_manager,
)


def _model_label() -> str:
    if cfg.BACKEND == "ollama":
        return f"ollama · {cfg.OLLAMA_MODEL}"
    return f"llama-cpp · {cfg.MODEL_PATH.name}"


def _save_memory(model: VibeModel):
    """Ask the model to summarise the session into .vibe/memory.md."""
    vibe_dir = Path.cwd() / ".vibe"
    vibe_dir.mkdir(exist_ok=True)
    mem_path = vibe_dir / "memory.md"

    existing = ""
    if mem_path.exists():
        try:
            existing = mem_path.read_text(encoding="utf-8").strip()
        except Exception:
            pass

    summary_prompt = (
        "Summarise this session concisely for future context. "
        "Include: key decisions, file layout, current status, any bugs/blockers. "
        "Output ONLY the markdown summary — no preamble, no fences.\n"
    )
    if existing:
        summary_prompt += f"\nPrevious memory:\n{existing}\n\nMerge with new info."

    # Use tool-free summarization to prevent accidental file modifications
    print_info("Summarizing session...")
    tokens = []
    try:
        for tok in model.summarize(summary_prompt):
            tokens.append(tok)
    except Exception as e:
        print_error(f"Summarization failed: {e}")
        return

    summary = "".join(tokens).strip()
    # Strip thinking blocks if present
    import re
    summary = re.sub(r"<think>[\s\S]*?</think>", "", summary).strip()
    if summary:
        mem_path.write_text(summary + "\n", encoding="utf-8")
        print_info(f"Session saved to {mem_path}")
    else:
        print_error("Empty summary — nothing saved.")


def _show_memory():
    mem_path = Path.cwd() / ".vibe" / "memory.md"
    if mem_path.exists():
        content = mem_path.read_text(encoding="utf-8").strip()
        if content:
            console.print(Panel(content, title=".vibe/memory.md", border_style="dim", box=box.ROUNDED))
        else:
            print_info("Memory file is empty.")
    else:
        print_info("No memory file yet. Use /save to create one.")


def _show_tokens(model: VibeModel):
    used = model.token_count()
    if cfg.BACKEND == "ollama":
        limit = cfg.OLLAMA_CTX
    else:
        limit = cfg.N_CTX
    pct = min(used / limit, 1.0) if limit else 0
    bar_len = 30
    filled = int(bar_len * pct)
    bar = "█" * filled + "░" * (bar_len - filled)
    console.print(f"  [bold]Context:[/] [{bar}] {used:,} / {limit:,} tokens ({pct:.0%})")
    if pct > 0.8:
        console.print("  [yellow]Warning: context is getting full. Consider /reset or /save then /reset.[/]")


def _handle_set(args: str):
    parts = args.strip().split(None, 1)
    if len(parts) < 2:
        print_error("Usage: /set <param> <value>  (temp, tokens, top_p, top_k, repeat_penalty)")
        return
    param, val = parts[0].lower(), parts[1]
    try:
        if param == "temp":
            cfg.TEMPERATURE = max(0.0, min(2.0, float(val)))
            print_info(f"temperature = {cfg.TEMPERATURE}")
        elif param == "tokens":
            cfg.MAX_TOKENS = max(1, int(val))
            print_info(f"max_tokens = {cfg.MAX_TOKENS}")
        elif param == "top_p":
            cfg.TOP_P = max(0.0, min(1.0, float(val)))
            print_info(f"top_p = {cfg.TOP_P}")
        elif param == "top_k":
            cfg.TOP_K = max(1, int(val))
            print_info(f"top_k = {cfg.TOP_K}")
        elif param == "repeat_penalty":
            cfg.REPEAT_PENALTY = max(0.0, float(val))
            print_info(f"repeat_penalty = {cfg.REPEAT_PENALTY}")
        else:
            print_error(f"Unknown param '{param}'. Options: temp, tokens, top_p, top_k, repeat_penalty")
    except ValueError:
        print_error(f"Invalid value: {val}")


def main():
    verbose = "--verbose" in sys.argv

    # Optional: set working directory from first positional arg
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        target = Path(args[0]).expanduser().resolve()
        if target.is_dir():
            os.chdir(target)
        else:
            print_error(f"Not a directory: {target}")
            sys.exit(1)

    try:
        model = VibeModel(verbose=verbose)
    except (FileNotFoundError, RuntimeError) as e:
        print_error(str(e))
        sys.exit(1)

    print_welcome(_model_label())

    while True:
        try:
            text = get_input(os.getcwd())
        except KeyboardInterrupt:
            console.print("\n[bold red]Interrupted.[/]")
            break

        if text is None:
            break
        if not text:
            continue

        # ── Slash commands ───────────────────────────────────────────
        if text.startswith("/"):
            cmd = text.split()[0].lower()
            rest = text[len(cmd):].strip()

            if cmd == "/exit":
                console.print("[dim]Goodbye![/]")
                break
            elif cmd == "/help":
                print_help()
            elif cmd == "/reset":
                model.reset()
                print_info("Conversation cleared.")
            elif cmd == "/think":
                cfg.THINKING = True
                print_info("Thinking mode ON — chain-of-thought enabled.")
            elif cmd == "/nothink":
                cfg.THINKING = False
                print_info("Thinking mode OFF — faster responses.")
            elif cmd == "/model":
                console.print(f"  Backend: {cfg.BACKEND}")
                if cfg.BACKEND == "ollama":
                    console.print(f"  Model:   {cfg.OLLAMA_MODEL}")
                    console.print(f"  Host:    {cfg.OLLAMA_HOST}")
                    console.print(f"  Context: {cfg.OLLAMA_CTX:,}")
                else:
                    console.print(f"  Model:   {cfg.MODEL_PATH.name}")
                    console.print(f"  Context: {cfg.N_CTX:,}")
                console.print(f"  Temp:    {cfg.TEMPERATURE}")
                console.print(f"  Tokens:  {cfg.MAX_TOKENS:,}")
                console.print(f"  Think:   {'on' if cfg.THINKING else 'off'}")
            elif cmd == "/tokens":
                _show_tokens(model)
            elif cmd == "/save":
                _save_memory(model)
            elif cmd == "/memory":
                _show_memory()
            elif cmd == "/sys":
                console.print(sys_info())
            elif cmd == "/gpu":
                console.print(gpu_info())
            elif cmd == "/net":
                console.print(net_info())
            elif cmd == "/ps":
                console.print(ps_list(rest or None))
            elif cmd == "/kill":
                if not rest:
                    print_error("Usage: /kill <pid|name>")
                else:
                    console.print(kill_process(rest))
            elif cmd == "/files":
                console.print(open_file_manager(rest or "."))
            elif cmd == "/service":
                parts = rest.split(None, 1)
                if not parts:
                    print_error("Usage: /service <name> [action]")
                else:
                    name = parts[0]
                    action = parts[1] if len(parts) > 1 else "status"
                    console.print(service_control(name, action))
            elif cmd == "/services":
                console.print(services_list(rest or None))
            elif cmd == "/set":
                _handle_set(rest)
            else:
                print_error(f"Unknown command: {cmd}. Type /help for commands.")
            continue

        # ── Chat ─────────────────────────────────────────────────────
        response = stream_response(model.chat(text))

        # Context warning
        used = model.token_count()
        limit = cfg.OLLAMA_CTX if cfg.BACKEND == "ollama" else cfg.N_CTX
        if limit and used / limit > 0.85:
            console.print(
                f"\n[yellow]⚠ Context {used:,}/{limit:,} ({used/limit:.0%}) — "
                f"consider /save then /reset[/]"
            )


if __name__ == "__main__":
    main()
