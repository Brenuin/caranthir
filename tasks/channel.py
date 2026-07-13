"""Append-only log connecting a running task-agent to whatever is watching it.

This is deliberately separate from the task-agent's own durable memory
(memory/agents/<namespace>/). The channel carries transient, moment-to-moment
signal — "here's what I'm doing right now," "here's my final answer" — for
whoever is polling it live. The task-agent's memory carries what it has
learned and should still know next time it's asked to do something.

Kept in-process (a plain list behind a lock-free append, since CPython's GIL
makes list.append/pop atomic enough for this single-writer/single-reader
use). If a future need arises for the channel itself to survive a process
restart, that's a reason to revisit — not a reason to add that complexity now.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Literal

EntryKind = Literal["progress", "report", "error"]


@dataclass
class ChannelEntry:
    id: int
    task_id: str
    kind: EntryKind
    text: str
    created_at: float
    read: bool = False


class SharedTaskChannel:
    """One channel per running task-agent invocation, keyed by task_id.

    Multiple task_ids can share one channel instance; entries are filtered
    by task_id on read so callers only see what's relevant to them.
    """

    def __init__(self) -> None:
        self._entries: list[ChannelEntry] = []
        self._ids = itertools.count(1)

    def publish(self, task_id: str, kind: EntryKind, text: str) -> ChannelEntry:
        entry = ChannelEntry(
            id=next(self._ids),
            task_id=task_id,
            kind=kind,
            text=text,
            created_at=time.time(),
        )
        self._entries.append(entry)
        return entry

    def unread(self, task_id: str) -> list[ChannelEntry]:
        return [e for e in self._entries if e.task_id == task_id and not e.read]

    def mark_read(self, entries: list[ChannelEntry]) -> None:
        ids = {e.id for e in entries}
        for entry in self._entries:
            if entry.id in ids:
                entry.read = True

    def all_for(self, task_id: str) -> list[ChannelEntry]:
        return [e for e in self._entries if e.task_id == task_id]

    def last_activity_at(self, task_id: str) -> float | None:
        entries = self.all_for(task_id)
        return max((e.created_at for e in entries), default=None)


# Process-wide default channel. A single voice session only ever has one
# active delegated task at a time (see decommission/lifecycle design), so a
# module-level singleton is simpler than threading an instance everywhere.
_default_channel = SharedTaskChannel()


def get_channel() -> SharedTaskChannel:
    return _default_channel
