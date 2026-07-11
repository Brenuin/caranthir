# Caranthir

Terminal agent with streaming replies, conversation memory, and tools. OpenAI and Anthropic via LangGraph.

## Layout

| File | Role |
|------|------|
| `main.py` | CLI, LLM/provider setup, LangGraph agent, streaming loop |
| `terminal_ui.py` | Banner, colors, incremental stream printer |
| `prompts.py` | System prompt |
| `hotreload/` | Dev hot reload: backend edits apply on the next turn, no restart |
| `tests.py` | Unit + optional live API tests |
| `.env` | API keys and defaults (not committed) |

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Put keys in `.env`:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_MODEL=gpt-4.1-mini          # optional default
CARANTHIR_PROVIDER=openai          # optional; else inferred from model name
```

## Run

```powershell
python main.py
python main.py --model claude-sonnet-4-5
python main.py --model claude-sonnet-4-5 --effort high --show-thinking
python main.py --model claude-sonnet-4-5 --hosted-tools
```

| Flag | Notes |
|------|--------|
| `--model` | Model id (default: `OPENAI_MODEL` or `gpt-4.1-mini`) |
| `--provider` | `openai` \| `anthropic` (inferred from name if omitted) |
| `--effort` | Anthropic reasoning effort: `low`…`max` |
| `--show-thinking` | Stream summarized thinking (Anthropic) |
| `--hosted-tools` | Anthropic web search + code execution |

Type a prompt and Enter. `/quit`, `/exit`, or `Ctrl+C` to leave.

## Behavior

- **Graph:** `assistant` ↔ local `ToolNode`; memory via `MemorySaver` (`thread_id=terminal`)
- **Local tools:** `get_current_time`
- **Hosted tools** (`--hosted-tools`, Anthropic only): web search, code execution
- **Streaming:** token output, thinking blocks, and tool markers as they arrive
- **Hot reload:** edits to `prompts.py`, `terminal_ui.py`, or agent logic in `main.py` are picked up on the next message; the REPL loop and CLI flags still need a restart

## Tests

```powershell
python tests.py              # unit + live API
python tests.py --skip-live  # unit only
```
