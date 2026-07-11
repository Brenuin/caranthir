"""Dev-only hot reload. See reloader.py for how it works and its limits."""

from hotreload.reloader import check_and_reload

__all__ = ["check_and_reload"]
