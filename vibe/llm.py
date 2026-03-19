import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Iterator

import vibe.config as cfg
from .tools import TOOL_SCHEMAS, execute_tool


SYSTEM_PROMPT = """\
You are Vibe, a local AI coding assistant. You help with software engineering tasks: \
writing code, debugging, refactoring, explaining code, and running commands.

You have tools to read/write/edit files, run shell commands, search code, and list directories. \
Use them freely and proactively.

Guidelines:
- Always read a file before editing it.
- Use write_file or edit_file to create/modify files — NEVER use bash echo/cat to write file content.
- Prefer edit_file over write_file for targeted changes. If edit_file returns "old_string not found", STOP and use write_file to rewrite the whole file — never retry edit_file with the same or similar string.
- Run tests after making changes when tests exist.
- Be concise — lead with action, not explanation.
- NEVER narrate what you are about to do before doing it. Do not say "I'll write the file now" or "Let me create..." — just call the tool immediately. Text before the first tool call is wasted tokens.
- If a task requires writing code, call write_file first. Explain afterward if needed.
- The current working directory is: {cwd}
- OS: Arch Linux. Package manager is pacman (or yay for AUR). Do NOT use apt/apt-get/brew.
- Python packages: install with pip inside the active virtualenv, not system pip.
- The bash tool runs in a subprocess with NO TTY. Do NOT run interactive or terminal-UI programs (curses, pygame, ncurses, etc.) through bash — they will always fail with errors like "cbreak() returned ERR". Write the code and tell the user to run it themselves.
- Games, TUIs, and any script with a game_loop/event_loop/main_loop CANNOT be tested via bash. They will always time out. When writing such programs: verify correctness by carefully reading the code, then tell the user to run it. NEVER attempt to run a game or interactive script through bash to "test" it.
- If bash returns a timeout error: STOP immediately. Do NOT retry, do NOT add timeout mechanisms to the script, do NOT keep editing and re-running. A timeout means the program needs a TTY. Tell the user to run it directly.
- GUI applications (file managers, text editors, browsers, etc.) CAN be launched through bash — the display environment (DISPLAY/WAYLAND_DISPLAY) is inherited. Launch them detached: `nohup thunar . &>/dev/null &`. Use xdg-open for generic file/URL opening. To open a file manager: try thunar, nautilus, dolphin, nemo, or pcmanfm in that order.
- Project memory is stored in .vibe/memory.md — read it at the start of a session if it exists, \
and update it with important decisions, file layouts, and current status so future sessions \
don't need to rediscover everything.
{memory_section}"""


class VibeModel:
    def __init__(self, verbose: bool = False):
        self._messages: list[dict] = []
        self._think_filter = _ThinkFilter()
        self._llm = None

        if cfg.BACKEND == "ollama":
            self._init_ollama()
        else:
            self._init_llama_cpp(verbose)

        self._reset_system()

    def _init_llama_cpp(self, verbose: bool):
        from llama_cpp import Llama
        if not cfg.MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {cfg.MODEL_PATH}\n"
                "Run ./setup.sh to download it."
            )
        self._llm = Llama(
            model_path=str(cfg.MODEL_PATH),
            n_gpu_layers=cfg.N_GPU_LAYERS,
            n_ctx=cfg.N_CTX,
            n_threads=cfg.N_THREADS,
            flash_attn=cfg.FLASH_ATTN,
            type_k=cfg.KV_CACHE_TYPE,
            type_v=cfg.KV_CACHE_TYPE,
            verbose=verbose,
        )

    def _init_ollama(self):
        try:
            urllib.request.urlopen(f"{cfg.OLLAMA_HOST}/api/tags", timeout=5)
        except Exception as e:
            raise RuntimeError(
                f"Ollama not reachable at {cfg.OLLAMA_HOST}: {e}\n"
                "Make sure ollama is running: ollama serve"
            )

    def _reset_system(self):
        cwd = os.getcwd()
        memory_path = Path(cwd) / ".vibe" / "memory.md"
        memory_section = ""
        if memory_path.exists():
            content = memory_path.read_text(encoding="utf-8").strip()
            if content:
                memory_section = f"\n## Memory from previous sessions\n{content}\n"
        self._messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(cwd=cwd, memory_section=memory_section)}
        ]

    def reset(self):
        self._think_filter = _ThinkFilter()
        self._reset_system()

    def token_count(self) -> int:
        text = " ".join(m.get("content", "") or "" for m in self._messages)
        if cfg.BACKEND == "llama_cpp" and self._llm:
            return len(self._llm.tokenize(text.encode())) if text else 0
        # ollama: rough estimate (4 chars ≈ 1 token)
        return len(text) // 4

    def _user_content(self, text: str) -> str:
        """Prepend Qwen3 thinking-mode directive to user messages."""
        directive = "" if cfg.THINKING else "/no_think "
        return directive + text

    def chat(self, user_text: str) -> Iterator[str]:
        """
        Send a user message and yield text tokens as they stream.
        Handles the full tool-call agentic loop internally.
        Yields special markers:
          - "\x00TOOL_START\x00{name}\x00{json_args}\x00"  — tool about to run
          - "\x00TOOL_END\x00{result}\x00"                 — tool result
        """
        self._messages.append({
            "role": "user",
            "content": self._user_content(user_text),
        })

        # Reset think filter at the start of each turn so interrupted
        # streams don't leave it in a bad state
        self._think_filter = _ThinkFilter()

        _autopush_remaining = 2  # max automatic nudges per user turn

        while True:
            # ── Call the model ────────────────────────────────────────────────
            stream = self._stream_completion()

            # ── Collect streamed response ─────────────────────────────────────
            assistant_text = ""
            tool_calls_acc: dict[int, dict] = {}

            # Rolling buffer: hold back enough chars to detect "<tool_call>"
            # before yielding, so we never stream that tag to the UI.
            _TAG = "<tool_call>"
            stream_buf = ""
            text_tool_call_started = False

            for chunk in stream:
                delta = chunk["choices"][0].get("delta", {})

                # Accumulate text tokens
                if delta.get("content"):
                    token = delta["content"]
                    assistant_text += token

                    if not text_tool_call_started:
                        stream_buf += token
                        if _TAG in stream_buf:
                            # Yield everything before the tag, then stop
                            pre = stream_buf[:stream_buf.find(_TAG)]
                            if pre:
                                yield from self._emit_text(pre)
                            text_tool_call_started = True
                            stream_buf = ""
                        elif len(stream_buf) > len(_TAG):
                            # Safe to yield all but last len(_TAG)-1 chars
                            safe_len = len(stream_buf) - len(_TAG) + 1
                            yield from self._emit_text(stream_buf[:safe_len])
                            stream_buf = stream_buf[safe_len:]
                    # else: text tool call in progress — accumulate only

                # Accumulate structured tool calls (when llama-cpp does parse them)
                if delta.get("tool_calls"):
                    for tc in delta["tool_calls"]:
                        idx = tc.get("index", 0)
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": tc.get("id", ""),
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        acc = tool_calls_acc[idx]
                        if tc.get("id"):
                            acc["id"] = tc["id"]
                        fn = tc.get("function", {})
                        if fn.get("name"):
                            acc["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            acc["function"]["arguments"] += fn["arguments"]

            # Flush any remaining buffered text (no tool_call detected)
            if stream_buf and not text_tool_call_started:
                yield from self._emit_text(stream_buf)

            tool_calls = [tool_calls_acc[i] for i in sorted(tool_calls_acc)]

            # Fallback: parse <tool_call> blocks from text when llama-cpp
            # streaming didn't populate delta["tool_calls"] (Qwen3 quirk)
            if not tool_calls and text_tool_call_started:
                tool_calls = _parse_text_tool_calls(assistant_text)

            # ── No tool calls → done ──────────────────────────────────────────
            if not tool_calls:
                # Check for stall: empty response, think-only, or narration without action
                _visible = re.sub(r"<think>[\s\S]*?</think>", "", assistant_text).strip()
                _is_stall = (
                    not _visible
                    or len(_visible) < 30
                    or _STALL_RE.search(_visible)
                )
                if _autopush_remaining > 0 and _is_stall:
                    _autopush_remaining -= 1
                    if assistant_text.strip():
                        self._messages.append({
                            "role": "assistant",
                            "content": assistant_text,
                        })
                    self._messages.append({
                        "role": "user",
                        "content": "Use write_file to do this now. No text, just the tool call.",
                    })
                    self._think_filter = _ThinkFilter()
                    continue

                self._messages.append({
                    "role": "assistant",
                    "content": assistant_text,
                })
                return

            # ── Execute tool calls ────────────────────────────────────────────
            self._messages.append({
                "role": "assistant",
                "content": assistant_text or None,
                "tool_calls": tool_calls,
            })

            for tc in tool_calls:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}

                yield f"\x00TOOL_START\x00{name}\x00{json.dumps(args)}\x00"

                result = execute_tool(name, args)

                yield f"\x00TOOL_END\x00{result}\x00"

                self._messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

            # Loop: send tool results back to model

    def _stream_completion(self):
        if cfg.BACKEND == "ollama":
            return self._ollama_stream()
        return self._llm.create_chat_completion(
            messages=self._messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=cfg.TEMPERATURE,
            top_p=cfg.TOP_P,
            top_k=cfg.TOP_K,
            min_p=cfg.MIN_P,
            repeat_penalty=cfg.REPEAT_PENALTY,
            max_tokens=cfg.MAX_TOKENS,
            stream=True,
        )

    def _ollama_stream(self):
        # Ollama does not emit tool_calls in streaming deltas (they appear only
        # in non-streaming responses). Use non-streaming and simulate a stream.
        payload = json.dumps({
            "model": cfg.OLLAMA_MODEL,
            "messages": self._messages,
            "tools": TOOL_SCHEMAS,
            "tool_choice": "auto",
            "stream": False,
            "temperature": cfg.TEMPERATURE,
            "top_p": cfg.TOP_P,
            "top_k": cfg.TOP_K,
            "repeat_penalty": cfg.REPEAT_PENALTY,
            "num_predict": cfg.MAX_TOKENS,
        }).encode()
        req = urllib.request.Request(
            f"{cfg.OLLAMA_HOST}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read())

        message = result["choices"][0]["message"]
        reasoning = (message.get("reasoning") or "").strip()
        content = (message.get("content") or "").strip()
        tool_calls = message.get("tool_calls") or []

        # Yield reasoning wrapped in think tags so the existing filter handles it
        if reasoning:
            chunk_text = f"<think>{reasoning}</think>"
            yield {"choices": [{"delta": {"content": chunk_text}, "finish_reason": None}]}

        # Yield content in small chunks for a streaming feel
        if content:
            for i in range(0, len(content), 20):
                yield {"choices": [{"delta": {"content": content[i:i+20]}, "finish_reason": None}]}

        # Yield tool calls as a single delta
        if tool_calls:
            yield {"choices": [{"delta": {"content": "", "tool_calls": tool_calls}, "finish_reason": "tool_calls"}]}

    def _emit_text(self, text: str) -> Iterator[str]:
        """Yield text through the think filter (or raw if thinking is on)."""
        if not cfg.THINKING:
            yield from self._think_filter.feed_iter(text)
        else:
            yield text


# ── Text tool-call parser ───────────────────────────────────────────────────────

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

# Detects "I'll write / Let me create / I will implement..." without any tool calls
_STALL_RE = re.compile(
    r"(?i)\b(i'?ll|let me|i will|i(?:'m| am) going to|i can|here(?:'s| is)(?: the| a)?)\b"
    r"[\s\S]{0,120}\b(write|create|build|implement|generate|code|make|develop|add)\b"
)


def _parse_text_tool_calls(text: str) -> list[dict]:
    """Extract Qwen3-style <tool_call>...</tool_call> blocks from assistant text."""
    calls = []
    for i, m in enumerate(_TOOL_CALL_RE.finditer(text)):
        try:
            data = json.loads(m.group(1))
            calls.append({
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": data.get("name", ""),
                    "arguments": json.dumps(data.get("arguments", {})),
                },
            })
        except json.JSONDecodeError:
            pass
    return calls


# ── Thinking-block stream filter ───────────────────────────────────────────────

class _ThinkFilter:
    """Stateful filter that strips <think>...</think> from a token stream."""
    def __init__(self):
        self._in_think = False
        self._buf = ""

    def feed_iter(self, token: str) -> Iterator[str]:
        self._buf += token
        while True:
            if self._in_think:
                end = self._buf.find("</think>")
                if end == -1:
                    self._buf = self._buf[-len("</think>"):]  # keep tail for partial match
                    return
                self._buf = self._buf[end + len("</think>"):]
                self._in_think = False
            else:
                start = self._buf.find("<think>")
                if start == -1:
                    out = self._buf
                    self._buf = ""
                    if out:
                        yield out
                    return
                out = self._buf[:start]
                self._buf = self._buf[start + len("<think>"):]
                self._in_think = True
                if out:
                    yield out
