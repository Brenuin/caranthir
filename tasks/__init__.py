"""The task-agent: a persistent, namespaced background worker Caranthir can delegate to.

See tasks/agent.py for why this is a distinct graph rather than a clone of
Caranthir's own build_agent(), and tasks/channel.py for how it reports back.
"""

from tasks.agent import build_task_agent, run_task
from tasks.channel import SharedTaskChannel, get_channel

__all__ = ["SharedTaskChannel", "build_task_agent", "get_channel", "run_task"]
