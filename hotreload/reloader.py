"""Hot reload: pick up backend code changes without restarting the REPL.

main.py calls check_and_reload() once per turn, right after you submit a
prompt. If any watched source file changed on disk, every watched module is
re-imported and the fresh `main` module is returned so the caller can rebuild
the agent from the new code. Conversation history survives because the
checkpointer lives in the running loop and is handed back to the rebuilt
agent.

What reloads: prompts.py, terminal_ui.py, and the agent logic in main.py
(tools, build_llm, build_agent, stream_turn, StreamPrinter).

What does NOT reload: the REPL loop itself and parse_args in main.py — the
already-running loop keeps executing its original code, and CLI flags were
parsed at startup. Changes there still need a restart. If reloading ever
leaves things in a weird state, just restart; this is a dev convenience,
not a guarantee.
"""

import importlib
import os
import sys

# Reloaded in this order when any file changes: dependencies first, so that
# when a dependent module re-executes its `from x import y` lines it binds
# the freshly reloaded objects.
WATCHED_MODULES = ["prompts", "terminal_ui", "main"]

_mtimes: dict[str, float] = {}


def _module(name: str):
    # When running `python main.py`, the script itself is `__main__`, not
    # `main` — this imports the file a second time under its real name so it
    # can be reloaded. Harmless: the `if __name__ == "__main__"` guard keeps
    # the second copy from starting another REPL.
    mod = sys.modules.get(name)
    if mod is None:
        mod = importlib.import_module(name)
    return mod


def _files_changed() -> bool:
    changed = False
    for name in WATCHED_MODULES:
        try:
            mtime = os.path.getmtime(_module(name).__file__)
        except OSError:
            continue  # editor mid-save; check again next turn
        previous = _mtimes.get(name)
        _mtimes[name] = mtime
        if previous is not None and mtime != previous:
            changed = True
    return changed


def check_and_reload():
    """Reload all watched modules if any source file changed since last call.

    Returns the fresh `main` module after a reload, or None when nothing
    changed. If the new code fails to import (e.g. a syntax error), the old
    modules stay in place, the error is printed, and None is returned — the
    running agent keeps working until the file is fixed.
    """
    if not _files_changed():
        return None
    try:
        for name in WATCHED_MODULES:
            importlib.reload(_module(name))
    except Exception as exc:
        print(f"  ! hot reload failed, still running old code: {exc}")
        return None
    return sys.modules["main"]
