import argparse
import os
from typing import Iterable

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph


DEFAULT_MODEL = "gpt-4.1-mini"
SYSTEM_PROMPT = (
    "You are Caranthir, a pragmatic terminal agent. "
    "Answer directly, ask for missing requirements when needed, and keep responses concise."
)


def build_agent(model: str):
    llm = ChatOpenAI(model=model)

    def assistant(state: MessagesState) -> dict[str, list[BaseMessage]]:
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        response = llm.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("assistant", assistant)
    graph.add_edge(START, "assistant")
    graph.add_edge("assistant", END)

    return graph.compile(checkpointer=MemorySaver())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Caranthir in the terminal.")
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        help=f"OpenAI chat model to use. Defaults to OPENAI_MODEL or {DEFAULT_MODEL}.",
    )
    return parser.parse_args()


def last_ai_text(messages: Iterable[BaseMessage]) -> str:
    for message in reversed(list(messages)):
        if isinstance(message, AIMessage):
            return message.content if isinstance(message.content, str) else str(message.content)
    return ""


def main() -> None:
    load_dotenv()
    args = parse_args()
    agent = build_agent(args.model)
    config = {"configurable": {"thread_id": "terminal"}}

    print(f"Caranthir is ready on {args.model}. Type /quit to exit.\n")

    while True:
        try:
            prompt = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if prompt.lower() in {"/quit", "/exit"}:
            break

        if not prompt:
            continue

        result = agent.invoke(
            {"messages": [HumanMessage(content=prompt)]},
            config=config,
        )

        print(f"\nCaranthir> {last_ai_text(result['messages'])}\n")


if __name__ == "__main__":
    main()
