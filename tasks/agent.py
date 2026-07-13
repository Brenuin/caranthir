"""The task-agent: a persistent, namespaced background worker.

Explicitly NOT a clone of Caranthir. It is a leaf — it has no delegate_task
tool of its own and cannot spawn further sub-agents — with its own system
prompt, own tool set, and its own namespaced memory/checkpoint files (see
memory.agent_data_dir). It never talks to the user; it only ever writes into
a SharedTaskChannel for whatever is watching (see tasks/channel.py), and its
final message on completion is treated as its report.

Caranthir's own build_agent() in main.py is a separate, differently-shaped
graph optimized for live conversation. This module intentionally does not
import from main.py, to keep the dependency direction one-way: main.py (and
voice/) depend on tasks/, not the other way around.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from memory import agent_data_dir, open_checkpointer
from tasks.channel import SharedTaskChannel, get_channel
from tasks.prompts import TASK_AGENT_SYSTEM_PROMPT

NAMESPACE = "task_agent"


def build_task_agent(
    llm: BaseChatModel,
    tools: list,
    task_id: str,
    channel: SharedTaskChannel | None = None,
    checkpointer=None,
):
    """Build the task-agent's graph.

    llm: an already-constructed chat model (caller decides provider/model —
    reuse main.build_llm() for this rather than duplicating that logic here).
    tools: the task-agent's own tool set. Deliberately passed in rather than
    hardcoded, since what a background worker should be allowed to do is a
    decision for the caller (main.py), not this module.
    task_id: identifies this run in the channel; also doubles as the
    checkpointer thread_id so a given task's history is addressable later.
    checkpointer: pass the namespaced one from agent_data_dir(NAMESPACE) to
    persist across runs; omit for a throwaway in-memory run (mainly for tests).
    """
    channel = channel or get_channel()
    bound_llm = llm.bind_tools(tools)

    def assistant(state: MessagesState) -> dict[str, list[BaseMessage]]:
        messages = [SystemMessage(content=TASK_AGENT_SYSTEM_PROMPT), *state["messages"]]
        response = bound_llm.invoke(messages)
        return {"messages": [response]}

    def report_progress(state: MessagesState) -> dict:
        # Runs after every assistant step. If this step produced tool calls,
        # that's progress worth surfacing; the *final* text-only step (no
        # more tool calls) is the report, published once the graph reaches END.
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            names = ", ".join(c["name"] for c in last.tool_calls)
            channel.publish(task_id, "progress", f"working: {names}")
        return {}

    graph = StateGraph(MessagesState)
    graph.add_node("assistant", assistant)
    graph.add_node("progress", report_progress)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "assistant")
    graph.add_edge("assistant", "progress")
    graph.add_conditional_edges("progress", tools_condition)
    graph.add_edge("tools", "assistant")

    return graph.compile(checkpointer=checkpointer or open_checkpointer(agent_data_dir(NAMESPACE) / "checkpoints.sqlite"))


def run_task(agent, task_id: str, description: str, channel: SharedTaskChannel | None = None) -> str:
    """Run one delegated task to completion; returns and publishes the report.

    Synchronous — call via asyncio.to_thread from an async caller so it
    doesn't block a live event loop (e.g. the voice session's mic loop).
    """
    channel = channel or get_channel()
    config = {"configurable": {"thread_id": task_id}}
    try:
        result = agent.invoke({"messages": [HumanMessage(content=description)]}, config=config)
        report_text = result["messages"][-1].content
        if not isinstance(report_text, str):
            report_text = str(report_text)
        channel.publish(task_id, "report", report_text)
        return report_text
    except Exception as exc:
        error_text = f"Task failed: {exc}"
        channel.publish(task_id, "error", error_text)
        return error_text
