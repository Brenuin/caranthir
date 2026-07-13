"""Native LangGraph conversation persistence.

LangGraph's checkpointer is its built-in memory system: after every step it
snapshots the whole graph state (the message list) into the checkpointer.
MemorySaver keeps those snapshots in RAM; SqliteSaver writes the exact same
snapshots to a SQLite file instead, so the conversation survives restarts
and you can open the file and look at what was saved.

The database lands next to this file as checkpoints.sqlite (gitignored).
Run `python -m memory.inspect` to see its contents decoded; delete the file
to start with a blank history.
"""

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

DEFAULT_DB = Path(__file__).with_name("checkpoints.sqlite")

# Namespaced agents (e.g. the task-agent) get their own subdirectory instead
# of sharing Caranthir's top-level checkpoints.sqlite / memories.json.
AGENTS_DIR = Path(__file__).with_name("agents")


def agent_data_dir(namespace: str) -> Path:
    """Directory holding one namespaced agent's checkpoint + memory files.

    Created on first use. Caranthir itself doesn't use this — it keeps the
    original top-level DEFAULT_DB / MemoryStore.DEFAULT_PATH for backward
    compatibility with existing conversation history.
    """
    path = AGENTS_DIR / namespace
    path.mkdir(parents=True, exist_ok=True)
    return path


def open_checkpointer(path: Path | str = DEFAULT_DB) -> SqliteSaver:
    # check_same_thread=False because LangGraph may touch the checkpointer
    # from worker threads during streaming.
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn)


def open_async_checkpointer(path: Path | str = DEFAULT_DB):
    """Async counterpart of open_checkpointer, for agent.astream() callers.

    Same on-disk schema and file as the sync SqliteSaver, so voice sessions
    (async) and the text REPL (sync) share conversation history through the
    same thread_id. Returns an async context manager — the caller must keep
    it open for the lifetime of the async agent's use.
    """
    return AsyncSqliteSaver.from_conn_string(str(path))
