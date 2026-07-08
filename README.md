# Caranthir

Caranthir is a small terminal agent built with LangGraph and OpenAI.

## Why LangGraph First?

LangChain is useful for model adapters, tools, retrievers, and reusable chain pieces.
LangGraph is the better starting point for this project because an agent needs explicit
control flow: state, memory, tool routing, approval steps, planning, retries, and
multi-step execution.

This app uses both:

- `langgraph` owns the agent loop and conversation state.
- `langchain-openai` adapts OpenAI chat models into that graph.

## Setup

Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

The `.env` file stores:

```text
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4.1-mini
```

Keep `.env` private. It is ignored by git.

## Run

```powershell
python main.py
```

Type a prompt and press Enter. Use `/exit`, `/quit`, or `Ctrl+C` to leave.

To override the model for one run:

```powershell
python main.py --model gpt-4.1-mini
```

## Current Architecture

The first graph is intentionally small:

```text
START -> assistant -> END
```

The terminal loop sends user messages into the graph. LangGraph checkpoints the
conversation in memory by `thread_id`, and the assistant node calls the OpenAI model
through LangChain.

Good next places to extend this are:

- tools: shell, filesystem search, project inspection, web search
- state: goals, active task, files touched, tool results
- routing: decide whether to answer, call tools, plan, or ask the user
- persistence: swap in a durable LangGraph checkpointer
