"""Show what LangGraph's SqliteSaver actually saved. Run after a session:

    python -m memory.inspect            # decoded view + raw table stats
    python -m memory.inspect --raw      # also dump raw rows (undecoded)

Two views of the same file:
- raw: the SQLite tables LangGraph writes (checkpoints = one full state
  snapshot per step, writes = pending channel updates between steps).
- decoded: those snapshots read back through the checkpointer API, i.e.
  the message history the agent would resume with.
"""

import argparse
import sqlite3
import sys

from langchain_core.messages import BaseMessage

from memory.checkpoints import DEFAULT_DB, open_checkpointer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def describe_message(message: BaseMessage) -> str:
    kind = type(message).__name__
    text = getattr(message, "text", "")
    if not isinstance(text, str):  # very old langchain-core: .text was a method
        text = text()
    text = " ".join(str(text).split())
    if len(text) > 110:
        text = text[:110] + "…"
    extras = ""
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        names = ", ".join(call["name"] for call in tool_calls)
        extras = f"  [tool calls: {names}]"
    return f"{kind:>13}: {text}{extras}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", action="store_true", help="Also dump raw table rows.")
    args = parser.parse_args()

    if not DEFAULT_DB.exists():
        print(f"No database at {DEFAULT_DB} — run a session first.")
        return

    conn = sqlite3.connect(str(DEFAULT_DB))
    tables = [
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    ]

    print(f"database: {DEFAULT_DB}")
    print("\n-- raw tables --")
    for table in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        columns = [c[1] for c in conn.execute(f"PRAGMA table_info({table})")]
        print(f"{table}: {count} rows  ({', '.join(columns)})")

    if args.raw:
        for table in tables:
            print(f"\n-- raw rows: {table} (up to 5) --")
            for row in conn.execute(f"SELECT * FROM {table} LIMIT 5"):
                cells = [
                    (repr(c)[:80] + "…") if len(repr(c)) > 80 else repr(c)
                    for c in row
                ]
                print("  " + " | ".join(cells))

    threads = [
        row[0]
        for row in conn.execute("SELECT DISTINCT thread_id FROM checkpoints")
    ]
    conn.close()

    saver = open_checkpointer()
    for thread in threads:
        config = {"configurable": {"thread_id": thread}}
        checkpoints = list(saver.list(config))
        latest = saver.get_tuple(config)
        print(f"\n-- thread {thread!r}: {len(checkpoints)} checkpoints --")
        if latest is None:
            continue
        messages = latest.checkpoint.get("channel_values", {}).get("messages", [])
        print(f"latest checkpoint holds {len(messages)} messages:")
        for message in messages:
            print("  " + describe_message(message))


if __name__ == "__main__":
    main()
