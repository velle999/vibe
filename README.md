# Vibe Code

A local AI coding assistant powered by [llama.cpp](https://github.com/ggerganov/llama.cpp) + CUDA. Runs entirely on your machine — no API keys, no cloud.

## Features

- Agentic tool-call loop: reads files, writes files, edits files, runs bash commands, globs, greps
- Qwen3 thinking mode (chain-of-thought reasoning, toggleable at runtime)
- Streaming output with Rich UI
- Session memory via `/save` → `.vibe/memory.md`
- Context usage tracking with visual bar

## Requirements

- Arch Linux (setup script uses pacman)
- NVIDIA GPU with CUDA support (~12GB VRAM for 8B Q8_0 at 32k context)
- Python 3.12+

## Setup

```bash
bash setup.sh
```

This will:
1. Create a virtualenv at `.venv/`
2. Install CUDA toolkit if missing
3. Build `llama-cpp-python` with CUDA support
4. Download `Qwen3-8B-Q8_0.gguf` (~8.5GB) into `models/`

## Usage

```bash
# Activate venv and launch
./vibe.sh

# Or point it at a project directory
./vibe.sh ~/my-project

# Verbose mode (shows tracebacks)
./vibe.sh --verbose
```

## Slash Commands

**Conversation**

| Command    | Description                                       |
|------------|---------------------------------------------------|
| `/reset`   | Clear conversation history                        |
| `/think`   | Enable Qwen3 chain-of-thought reasoning           |
| `/nothink` | Disable chain-of-thought (faster)                 |
| `/tokens`  | Show context usage with a visual bar              |
| `/model`   | Show current model path and settings              |
| `/save`    | Summarize session to `.vibe/memory.md`            |
| `/memory`  | Print current `.vibe/memory.md`                   |
| `/exit`    | Quit                                              |
| `/help`    | Show all commands                                 |

**System Info**

| Command           | Description                                     |
|-------------------|-------------------------------------------------|
| `/sys`            | CPU, RAM, disk usage, uptime                    |
| `/gpu`            | GPU utilization, VRAM usage, temperature        |
| `/net`            | Network interfaces and listening ports          |
| `/ps [filter]`    | Top processes by CPU (optional name filter)     |

**Process Control**

| Command            | Description                                    |
|--------------------|------------------------------------------------|
| `/kill <pid\|name>` | Send SIGTERM to a PID or matching processes   |

**Service Control**

| Command                       | Description                              |
|-------------------------------|------------------------------------------|
| `/service <name> [action]`    | Control a systemd service (default: status). Actions: `start` `stop` `restart` `reload` `enable` `disable` |
| `/services [filter]`          | List running services (optional filter)  |

**Runtime Config**

| Command                    | Description                              |
|----------------------------|------------------------------------------|
| `/set temp <0.0-2.0>`      | Generation temperature                   |
| `/set tokens <n>`          | Max output tokens                        |
| `/set top_p <0.0-1.0>`     | Nucleus sampling probability             |
| `/set top_k <n>`           | Top-k sampling                           |
| `/set repeat_penalty <n>`  | Repetition penalty                       |

## Models

Default: `models/Qwen3-8B-Q8_0.gguf`

To use a different model, change `MODEL_PATH` in `vibe/config.py`.

## Project Structure

```
vibe-code/
├── main.py          # REPL entry point, slash command handling
├── vibe/
│   ├── config.py    # Model path, generation params, THINKING flag
│   ├── llm.py       # VibeModel class — agentic tool-call loop, thinking filter
│   ├── tools.py     # Tool schemas + implementations (read, write, edit, bash, glob, grep, ls)
│   └── ui.py        # Rich console UI, prompt_toolkit session, streaming renderer
├── setup.sh         # One-shot setup script
├── vibe.sh          # Launch wrapper
└── requirements.txt
```

## Configuration

Edit `vibe/config.py` to tune generation settings:

```python
N_CTX        = 32768   # context window (tokens)
N_GPU_LAYERS = -1      # GPU layers (-1 = all)
TEMPERATURE  = 0.6
MAX_TOKENS   = 8192
THINKING     = False   # chain-of-thought off by default
```
