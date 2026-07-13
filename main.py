import argparse
import asyncio
import os
from datetime import datetime
from typing import Iterable

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from document_rag import DOCUMENT_RAG_TOOLS
from hotreload import check_and_reload
from memory import MEMORY_TOOLS, memory_context, open_checkpointer
from prompts import SYSTEM_PROMPT
from terminal_ui import (
    StreamPrinter,
    print_banner,
    print_error,
    print_notice,
    read_user_input,
)


DEFAULT_MODEL = "gpt-4.1-mini"

# Model name prefix -> provider. Lets --model pick the provider implicitly;
# --provider overrides this when a name doesn't match any prefix below.
PROVIDER_PREFIXES = {
    "gpt-": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
    "claude-": "anthropic",
}


def infer_provider(model: str) -> str:
    lowered = model.lower()
    for prefix, provider in PROVIDER_PREFIXES.items():
        if lowered.startswith(prefix):
            return provider
    raise ValueError(
        f"Can't infer provider for model {model!r}. Pass --provider explicitly."
    )


EFFORT_LEVELS = {"low", "medium", "high", "xhigh", "max"}


def build_llm(
    model: str, provider: str | None, effort: str | None, show_thinking: bool = False
) -> tuple[BaseChatModel, str]:
    provider = provider or infer_provider(model)
    if provider == "openai":
        if effort:
            raise ValueError("--effort is only supported for Anthropic models right now.")
        if show_thinking:
            raise ValueError("--show-thinking is only supported for Anthropic models right now.")
        return ChatOpenAI(model=model), provider
    if provider == "anthropic":
        kwargs = {}
        if effort:
            kwargs["effort"] = effort
        if show_thinking:
            # Thinking happens either way on adaptive-thinking models; "display"
            # only controls whether the API returns a readable summary of it.
            kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
        return ChatAnthropic(model=model, **kwargs), provider
    raise ValueError(f"Unknown provider {provider!r}. Use 'openai' or 'anthropic'.")


@tool
def get_current_time() -> str:
    """Return the current local date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Sentinel string returned by start_voice() and checked for by the REPL loop
# after a turn finishes, since a tool can't switch I/O modes by itself — it
# can only signal the caller to do so.
VOICE_MODE_SENTINEL = "__ENTER_VOICE_MODE__"


@tool
def start_voice() -> str:
    """Switch the conversation to live voice mode (microphone + speakers).

    Call this when the user asks to talk out loud, switch to voice, or have
    a spoken conversation instead of typing — phrases like "let's talk over
    voice" or "can we speak instead of typing"."""
    return VOICE_MODE_SENTINEL


# Local tools: run on this machine via ToolNode.
TOOLS = [get_current_time, start_voice, *MEMORY_TOOLS, *DOCUMENT_RAG_TOOLS]

# Restricted tool set bound to the agent while a voice session is active:
# no hosted tools, no code execution — keeps voice turns fast and simple.
# start_voice is intentionally omitted; you're already in voice mode.
VOICE_TOOLS = [get_current_time, *MEMORY_TOOLS, *DOCUMENT_RAG_TOOLS]

# Anthropic server-side tools: executed on Anthropic's servers inside a single
# model call. They never appear in AIMessage.tool_calls, so tools_condition
# (correctly) ignores them and the graph needs no changes.
HOSTED_TOOLS = [
    {"type": "web_search_20260209", "name": "web_search"},
    {"type": "code_execution_20260521", "name": "code_execution"},
]


def build_agent(
    model: str,
    provider: str | None,
    effort: str | None,
    hosted: bool,
    show_thinking: bool = False,
    checkpointer=None,
    tools: list = None,
):
    # Pass an existing checkpointer to keep conversation history across
    # rebuilds (hot reload does this); omit it for a fresh one.
    # `tools` overrides the default TOOLS list (used for the voice-restricted
    # agent); hosted tools never apply there since voice callers pass hosted=False.
    local_tools = tools if tools is not None else TOOLS
    llm, resolved_provider = build_llm(model, provider, effort, show_thinking)
    if hosted and resolved_provider != "anthropic":
        raise ValueError("--hosted-tools is only supported for Anthropic models right now.")
    bindable = [*HOSTED_TOOLS, *local_tools] if hosted else local_tools
    llm = llm.bind_tools(bindable)

    def assistant(state: MessagesState) -> dict[str, list[BaseMessage]]:
        # memory_context() is re-read per turn so freshly saved facts apply
        # to the very next message.
        messages = [
            SystemMessage(content=SYSTEM_PROMPT + memory_context()),
            *state["messages"],
        ]
        response = llm.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("assistant", assistant)
    graph.add_node("tools", ToolNode(local_tools))
    graph.add_edge(START, "assistant")
    graph.add_conditional_edges("assistant", tools_condition)
    graph.add_edge("tools", "assistant")

    return graph.compile(checkpointer=checkpointer or MemorySaver()), resolved_provider


def build_voice_agent(model: str, provider: str | None, checkpointer=None):
    """Build an agent scoped to VOICE_TOOLS (no hosted tools, no start_voice).

    Uses the same checkpointer/thread as the text agent so conversation
    history is shared across text and voice turns.
    """
    agent, resolved_provider = build_agent(
        model,
        provider,
        effort=None,
        hosted=False,
        show_thinking=False,
        checkpointer=checkpointer,
        tools=VOICE_TOOLS,
    )
    return agent, resolved_provider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Caranthir in the terminal.")
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        help=f"Chat model to use. Defaults to OPENAI_MODEL or {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--provider",
        default=os.getenv("CARANTHIR_PROVIDER"),
        choices=["openai", "anthropic"],
        help="Force a provider instead of inferring it from --model's name.",
    )
    parser.add_argument(
        "--effort",
        default=None,
        choices=sorted(EFFORT_LEVELS),
        help="Enable Claude extended thinking at this reasoning effort (Anthropic only).",
    )
    parser.add_argument(
        "--hosted-tools",
        action="store_true",
        help="Enable Anthropic server-side web search and code execution (Anthropic only).",
    )
    parser.add_argument(
        "--show-thinking",
        action="store_true",
        help="Print the model's summarized reasoning above each reply (Anthropic only).",
    )
    return parser.parse_args()


def new_ai_text(messages: Iterable[BaseMessage], seen_ids: set[str]) -> str:
    """Collect text from every not-yet-printed AIMessage, in order.

    A turn with tool calls can produce several AIMessages, and models sometimes
    put substantive text in the pre-tool-call message; printing only the last
    one can silently drop part of the answer.
    """
    parts = []
    for message in messages:
        if not isinstance(message, AIMessage) or message.id in seen_ids:
            continue
        seen_ids.add(message.id)
        text = extract_text(message.content).strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def extract_text(content: str | list) -> str:
    if isinstance(content, str):
        return content
    parts = [
        block["text"]
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(parts)


def extract_thinking(content: str | list) -> str:
    if isinstance(content, str):
        return ""
    parts = [
        block["thinking"]
        for block in content
        if isinstance(block, dict) and block.get("type") == "thinking" and block.get("thinking")
    ]
    return "\n".join(parts)


def stream_turn(agent, prompt: str, config: dict, printer: StreamPrinter) -> bool:
    """Run one turn, printing output incrementally as it streams.

    stream_mode="messages" yields (chunk, metadata) pairs: AIMessageChunks
    token-by-token from any LLM call in the graph, and complete ToolMessages
    when local tools finish.

    Returns True if start_voice fired this turn, signaling main() to enter
    voice mode once the turn (and any remaining graph steps) finishes.
    """
    voice_requested = False
    for chunk, _metadata in agent.stream(
        {"messages": [HumanMessage(content=prompt)]},
        config=config,
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessageChunk):
            printer.feed_ai_content(chunk.content)
        elif isinstance(chunk, ToolMessage):
            printer.feed_local_tool(chunk.name, str(chunk.content))
            if chunk.name == "start_voice" and str(chunk.content) == VOICE_MODE_SENTINEL:
                voice_requested = True
    return voice_requested


def main() -> None:
    load_dotenv()
    args = parse_args()
    # SQLite-backed LangGraph checkpointer: conversation survives restarts
    # and hot reloads. Inspect with `python -m memory.inspect`.
    checkpointer = open_checkpointer()
    try:
        agent, resolved_provider = build_agent(
            args.model,
            args.provider,
            args.effort,
            args.hosted_tools,
            args.show_thinking,
            checkpointer=checkpointer,
        )
    except ValueError as exc:
        print_error(str(exc))
        return
    config = {"configurable": {"thread_id": "terminal"}}

    print_banner(args.model, resolved_provider)

    make_printer, run_turn = StreamPrinter, stream_turn
    while True:
        try:
            prompt = read_user_input().strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if prompt.lower() in {"/quit", "/exit"}:
            break

        if not prompt:
            continue

        fresh = check_and_reload()
        if fresh is not None:
            try:
                agent, resolved_provider = fresh.build_agent(
                    args.model,
                    args.provider,
                    args.effort,
                    args.hosted_tools,
                    args.show_thinking,
                    checkpointer=checkpointer,
                )
                make_printer, run_turn = fresh.StreamPrinter, fresh.stream_turn
                print_notice("code changed on disk — reloaded")
            except Exception as exc:
                print_error(f"reload failed, keeping old agent: {exc}")

        printer = make_printer("Caranthir", show_thinking=args.show_thinking)
        try:
            voice_requested = run_turn(agent, prompt, config, printer)
        except KeyboardInterrupt:
            printer.finish()
            print_error("interrupted — reply may be incomplete")
            continue
        except Exception as exc:
            printer.finish()
            print_error(str(exc))
            continue
        printer.finish()

        if voice_requested:
            from voice import run_voice_mode

            try:
                asyncio.run(
                    run_voice_mode(build_voice_agent, args.model, args.provider, config)
                )
            except SystemExit as exc:
                print_error(str(exc))
            except Exception as exc:
                print_error(f"voice session failed: {exc}")


if __name__ == "__main__":
    main()
