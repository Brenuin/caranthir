"""Agent-facing memory tools.

Saved memories are injected into the system prompt every turn, so the model
never needs a lookup tool — only save and delete. Docstrings here are what
the model reads when deciding to call a tool, so they carry the usage rules.
"""

from langchain_core.tools import tool

from memory.store import get_store


@tool
def remember(fact: str) -> str:
    """Save one short fact about your master to long-term memory so future
    conversations know it. Use for stable facts: their name, preferences,
    projects, people, standing decisions. Do not save chit-chat, secrets they
    asked you to keep out of writing, or things only true right now. Keep it
    to a single sentence."""
    entry = get_store().add(fact)
    return f"Saved memory [{entry['id']}]: {entry['text']}"


@tool
def forget(memory_id: int) -> str:
    """Delete one saved memory by the [id] shown in your long-term memory
    list. Use when your master corrects a saved fact or asks you to forget
    something; save the corrected version with remember afterwards."""
    if get_store().forget(memory_id):
        return f"Deleted memory [{memory_id}]."
    return f"No memory with id [{memory_id}]."


MEMORY_TOOLS = [remember, forget]
