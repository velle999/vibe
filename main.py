#!/usr/bin/env python3
"""Vibe Code — local AI coding assistant powered by llama.cpp + CUDA."""

import os
import sys
from pathlib import Path


def main():
    # ── Parse args ────────────────────────────────────────────────────────────
    args = sys.argv[1:]
    verbose = "--verbose" in args or "-v" in args
    args = [a for a in args if a not in ("--verbose", "-v")]

    # Optional: change working directory to a project path
    if args:
        target = Path(args[0]).expanduser().resolve()
        if target.is_dir():
            os.chdir(target)
            print(f"Working directory: {target}")
        else:
            print(f"Warning: {args[0]} is not a directory, using cwd")

    cwd = os.getcwd()

    # ── Load model ────────────────────────────────────────────────────────────
    from vibe.ui import console, print_welcome, print_help, print_info, print_error, get_input, stream_response
    import vibe.config as cfg

    print_info(f"Loading model… (this takes ~10s on first run)")
    try:
        from vibe.llm import VibeModel
        model = VibeModel(verbose=verbose)
    except FileNotFoundError as e:
        print_error(str(e))
        sys.exit(1)
    except Exception as e:
        print_error(f"Failed to load model: {e}")
        sys.exit(1)

    from vibe.system import (
        sys_info, gpu_info, net_info,
        ps_list, kill_process,
        service_control, services_list,
        open_file_manager,
    )

    print_welcome(str(cfg.MODEL_PATH))

    # ── REPL ──────────────────────────────────────────────────────────────────
    while True:
        user_input = get_input(cwd)

        if user_input is None:
            print_info("Bye!")
            break

        if not user_input:
            continue

        # Slash commands
        cmd = user_input.lower().strip()
        raw = user_input.strip()

        if cmd in ("/exit", "/quit", "exit", "quit"):
            print_info("Bye!")
            break

        if cmd == "/help":
            print_help()
            continue

        if cmd == "/reset":
            model.reset()
            print_info("Conversation reset.")
            continue

        if cmd == "/think":
            cfg.THINKING = True
            print_info("Chain-of-thought enabled.")
            continue

        if cmd == "/nothink":
            cfg.THINKING = False
            print_info("Chain-of-thought disabled.")
            continue

        if cmd == "/model":
            if cfg.BACKEND == "ollama":
                print_info(f"Backend: ollama  Model: {cfg.OLLAMA_MODEL}  Host: {cfg.OLLAMA_HOST}")
            else:
                print_info(f"Backend: llama_cpp  Model: {cfg.MODEL_PATH}")
                print_info(f"Context: {cfg.N_CTX} tokens  GPU layers: {cfg.N_GPU_LAYERS}")
            continue

        if cmd.startswith("/files"):
            parts = raw.split(None, 1)
            path = parts[1] if len(parts) > 1 else cwd
            print_info(open_file_manager(path))
            continue

        if cmd == "/sys":
            print_info(sys_info())
            continue

        if cmd == "/gpu":
            print_info(gpu_info())
            continue

        if cmd == "/net":
            print_info(net_info())
            continue

        if cmd.startswith("/ps"):
            parts = raw.split(None, 1)
            filter_str = parts[1] if len(parts) > 1 else None
            print_info(ps_list(filter_str))
            continue

        if cmd.startswith("/kill "):
            target = raw.split(None, 1)[1].strip()
            print_info(kill_process(target))
            continue

        if cmd.startswith("/service "):
            parts = raw.split()
            if len(parts) < 2:
                print_error("Usage: /service <name> [status|start|stop|restart|reload|enable|disable]")
            else:
                name = parts[1]
                action = parts[2] if len(parts) > 2 else "status"
                print_info(service_control(name, action))
            continue

        if cmd.startswith("/services"):
            parts = raw.split(None, 1)
            filter_str = parts[1] if len(parts) > 1 else None
            print_info(services_list(filter_str))
            continue

        if cmd.startswith("/set "):
            parts = raw.split()
            valid_params = {
                "temp": ("TEMPERATURE", float, 0.0, 2.0),
                "tokens": ("MAX_TOKENS", int, 64, 32768),
                "top_p": ("TOP_P", float, 0.0, 1.0),
                "top_k": ("TOP_K", int, 1, 200),
                "repeat_penalty": ("REPEAT_PENALTY", float, 1.0, 2.0),
            }
            if len(parts) < 3:
                opts = "  ".join(f"{k}" for k in valid_params)
                print_error(f"Usage: /set <param> <value>  — params: {opts}")
            else:
                param, val_str = parts[1].lower(), parts[2]
                if param not in valid_params:
                    opts = ", ".join(valid_params)
                    print_error(f"Unknown param '{param}'. Valid: {opts}")
                else:
                    attr, typ, lo, hi = valid_params[param]
                    try:
                        val = typ(val_str)
                        if not (lo <= val <= hi):
                            print_error(f"{param} must be between {lo} and {hi}")
                        else:
                            setattr(cfg, attr, val)
                            print_info(f"{param} = {val}")
                    except ValueError:
                        print_error(f"Invalid value '{val_str}' for {param} (expected {typ.__name__})")
            continue

        if cmd == "/tokens":
            used = model.token_count()
            pct = used / cfg.N_CTX * 100
            bar_filled = int(pct / 5)
            bar = "█" * bar_filled + "░" * (20 - bar_filled)
            color = "red" if pct > 80 else "yellow" if pct > 60 else "green"
            print_info(f"[{color}]{bar}[/] {used:,} / {cfg.N_CTX:,} tokens ({pct:.1f}%)")
            continue

        if cmd == "/memory":
            mem_path = Path(cwd) / ".vibe" / "memory.md"
            if mem_path.exists():
                print_info(mem_path.read_text())
            else:
                print_info("No memory file yet. Use /save to create one.")
            continue

        if cmd == "/save":
            # Use a fresh model call that won't hit context limits
            # by trimming history to just the save request
            save_prompt = (
                "Summarize this session into .vibe/memory.md — include: what we were building, "
                "key files and their purpose, decisions made, current status, and any known issues. "
                "Be concise but complete. Create the .vibe/ directory if needed."
            )
            try:
                token_stream = model.chat(save_prompt)
                stream_response(token_stream)
            except Exception as e:
                # Context may be full — trim to just the save request and retry
                try:
                    model.reset()
                    token_stream = model.chat(save_prompt + " (no prior context available — write a placeholder)")
                    stream_response(token_stream)
                except Exception as e2:
                    print_error(f"Save failed: {e2}")
            continue

        # Chat
        try:
            # Warn at 80% context usage
            used = model.token_count()
            pct = used / cfg.N_CTX * 100
            if pct >= 80:
                print_info(f"[yellow]Context {pct:.0f}% full — consider /save then /reset soon[/]")

            token_stream = model.chat(user_input)
            stream_response(token_stream)
        except KeyboardInterrupt:
            print_info("\nCancelled.")
        except Exception as e:
            msg = str(e)
            if "exceed context window" in msg or "exceed context" in msg.lower():
                used = model.token_count()
                print_error(f"Context full ({used:,}/{cfg.N_CTX:,} tokens). Use /reset to start fresh.")
            else:
                print_error(f"Generation failed: {e}")
            if verbose:
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
