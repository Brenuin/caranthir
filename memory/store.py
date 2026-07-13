"""File-backed long-term memory for Caranthir.

The checkpointer already gives the agent memory within a session; this layer
is for facts that should survive across sessions (quit, come back tomorrow).
Memories live in memories.json next to this file — human-readable, editable,
and deletable by hand if you ever want to inspect or wipe what it knows.

The file is re-read on every access, so there is no in-process cache to go
stale across hot reloads or concurrent edits.
"""

import json
from datetime import datetime
from pathlib import Path

DEFAULT_PATH = Path(__file__).with_name("memories.json")


class MemoryStore:
    def __init__(self, path: Path | str = DEFAULT_PATH):
        self.path = Path(path)

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        return data if isinstance(data, list) else []

    def _save(self, entries: list[dict]) -> None:
        self.path.write_text(
            json.dumps(entries, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def all(self) -> list[dict]:
        return self._load()

    def add(self, text: str) -> dict:
        text = " ".join(text.split())
        entries = self._load()
        for entry in entries:
            if entry["text"].lower() == text.lower():
                return entry  # already known; don't duplicate
        entry = {
            "id": max((e["id"] for e in entries), default=0) + 1,
            "text": text,
            "created": datetime.now().strftime("%Y-%m-%d"),
        }
        entries.append(entry)
        self._save(entries)
        return entry

    def forget(self, memory_id: int) -> bool:
        entries = self._load()
        kept = [e for e in entries if e["id"] != memory_id]
        if len(kept) == len(entries):
            return False
        self._save(kept)
        return True


_active = MemoryStore()


def get_store() -> MemoryStore:
    return _active


def set_store(store: MemoryStore) -> None:
    """Swap the active store; tests point this at a temp file."""
    global _active
    _active = store


def memory_context() -> str:
    """System-prompt block listing every saved memory, or "" when empty.

    Rebuilt each turn, so a fact saved mid-conversation is visible on the
    very next message.
    """
    entries = get_store().all()
    if not entries:
        return ""
    lines = "\n".join(f"- [{e['id']}] {e['text']} ({e['created']})" for e in entries)
    return (
        "\n\nLong-term memory — facts you saved in past conversations. "
        "Use them naturally; don't recite them unprompted:\n" + lines
    )
