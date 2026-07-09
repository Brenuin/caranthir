"""Test suite for Caranthir.

Run all tests:        python tests.py
Skip live API tests:  python tests.py --skip-live

Unit tests are free and instant. Live tests make real API calls
(OpenAI + Anthropic) and cost a small amount of tokens.
"""

import argparse
import time
import traceback

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from main import build_agent, build_llm, extract_text, infer_provider, new_ai_text
from terminal_ui import BOLD, DIM, GREEN, RED, style


RESULTS: list[tuple[str, bool, str]] = []


def run_test(name: str, fn) -> None:
    start = time.time()
    try:
        fn()
        elapsed = time.time() - start
        RESULTS.append((name, True, f"{elapsed:.1f}s"))
        print(f"{style('PASS', BOLD, GREEN)}  {name} {style(f'({elapsed:.1f}s)', DIM)}")
    except Exception as exc:
        elapsed = time.time() - start
        detail = f"{type(exc).__name__}: {exc}"
        RESULTS.append((name, False, detail))
        print(f"{style('FAIL', BOLD, RED)}  {name} {style(f'({elapsed:.1f}s)', DIM)}")
        print(style(f"      {detail}", RED))
        if not isinstance(exc, AssertionError):
            print(style(traceback.format_exc(limit=3), DIM))


# ---------------------------------------------------------------- unit tests


def test_infer_provider():
    assert infer_provider("gpt-4.1-mini") == "openai"
    assert infer_provider("o3-mini") == "openai"
    assert infer_provider("claude-fable-5") == "anthropic"
    assert infer_provider("CLAUDE-SONNET-5") == "anthropic"
    try:
        infer_provider("mystery-9000")
        raise AssertionError("expected ValueError for unknown model")
    except ValueError:
        pass


def test_effort_guard_openai():
    try:
        build_llm("gpt-4.1-mini", None, "high")
        raise AssertionError("expected ValueError for --effort on OpenAI")
    except ValueError as exc:
        assert "effort" in str(exc).lower()


def test_hosted_guard_openai():
    try:
        build_agent("gpt-4.1-mini", None, None, True)
        raise AssertionError("expected ValueError for --hosted-tools on OpenAI")
    except ValueError as exc:
        assert "hosted" in str(exc).lower()


def test_extract_text_string():
    assert extract_text("plain reply") == "plain reply"


def test_extract_text_blocks():
    content = [
        {"type": "thinking", "thinking": "secret reasoning", "signature": "xyz"},
        {"type": "text", "text": "visible answer"},
        {"type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {}},
        {"type": "text", "text": "second part"},
    ]
    assert extract_text(content) == "visible answer\nsecond part"
    assert "secret" not in extract_text(content)


def test_new_ai_text_collects_and_dedupes():
    m1 = AIMessage(content="first part", id="ai_1")
    m2 = AIMessage(content="final part", id="ai_2")
    messages = [HumanMessage(content="q"), m1, m2]
    seen: set[str] = set()
    assert new_ai_text(messages, seen) == "first part\n\nfinal part"
    # Second call with the same messages prints nothing new.
    assert new_ai_text(messages, seen) == ""
    # A new message after the dedupe still shows up.
    messages.append(AIMessage(content="turn two", id="ai_3"))
    assert new_ai_text(messages, seen) == "turn two"


# ---------------------------------------------------------------- live tests


def _one_turn(agent, prompt: str, thread_id: str):
    return agent.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config={"configurable": {"thread_id": thread_id}},
    )


def test_live_openai_local_tool():
    agent, provider = build_agent("gpt-4.1-mini", None, None, False)
    assert provider == "openai"
    result = _one_turn(agent, "What time is it? Use your tool.", "t-openai")
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs, "expected get_current_time to be called"
    assert tool_msgs[0].name == "get_current_time"
    assert new_ai_text(result["messages"], set()).strip(), "expected a text reply"


def test_live_anthropic_reply():
    agent, provider = build_agent("claude-sonnet-5", None, None, False)
    assert provider == "anthropic"
    result = _one_turn(agent, "Reply with exactly the word: pumpernickel", "t-claude")
    reply = new_ai_text(result["messages"], set())
    assert "pumpernickel" in reply.lower(), f"unexpected reply: {reply!r}"


def test_live_anthropic_memory():
    agent, _ = build_agent("claude-sonnet-5", None, None, False)
    _one_turn(agent, "My favorite color is vermilion. Remember it.", "t-memory")
    result = _one_turn(agent, "What is my favorite color? One word.", "t-memory")
    reply = new_ai_text(result["messages"], set())
    assert "vermilion" in reply.lower(), f"memory failed, reply: {reply!r}"


def test_live_effort_thinking():
    agent, _ = build_agent("claude-sonnet-5", None, "low", False)
    result = _one_turn(agent, "What is 17 * 23? Answer with just the number.", "t-effort")
    reply = new_ai_text(result["messages"], set())
    assert "391" in reply.replace(",", ""), f"expected 391 in reply: {reply!r}"


def test_live_hosted_code_execution():
    agent, _ = build_agent("claude-sonnet-5", None, None, True)
    result = _one_turn(
        agent, "Use your code tool to compute 2**20 exactly. State the number.", "t-hosted"
    )
    hosted_blocks = [
        block
        for m in result["messages"]
        if isinstance(m, AIMessage) and isinstance(m.content, list)
        for block in m.content
        if isinstance(block, dict) and block.get("type") == "server_tool_use"
    ]
    assert hosted_blocks, "expected a server_tool_use block from hosted code execution"
    reply = new_ai_text(result["messages"], set())
    assert "1048576" in reply.replace(",", ""), f"expected 1048576 in reply: {reply!r}"


# ---------------------------------------------------------------- runner


UNIT_TESTS = [
    ("unit: provider inference", test_infer_provider),
    ("unit: --effort rejected on OpenAI", test_effort_guard_openai),
    ("unit: --hosted-tools rejected on OpenAI", test_hosted_guard_openai),
    ("unit: extract_text on plain string", test_extract_text_string),
    ("unit: extract_text filters thinking blocks", test_extract_text_blocks),
    ("unit: new_ai_text collects and dedupes", test_new_ai_text_collects_and_dedupes),
]

LIVE_TESTS = [
    ("live: OpenAI local tool loop", test_live_openai_local_tool),
    ("live: Anthropic basic reply", test_live_anthropic_reply),
    ("live: Anthropic memory across turns", test_live_anthropic_memory),
    ("live: extended thinking (--effort)", test_live_effort_thinking),
    ("live: hosted code execution (--hosted-tools)", test_live_hosted_code_execution),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Caranthir's test suite.")
    parser.add_argument("--skip-live", action="store_true", help="Skip tests that call real APIs.")
    args = parser.parse_args()

    load_dotenv()

    print(style("\n-- unit tests --", BOLD))
    for name, fn in UNIT_TESTS:
        run_test(name, fn)

    if args.skip_live:
        print(style("\n-- live tests skipped (--skip-live) --", DIM))
    else:
        print(style("\n-- live tests (real API calls) --", BOLD))
        for name, fn in LIVE_TESTS:
            run_test(name, fn)

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = len(RESULTS) - passed
    color = GREEN if failed == 0 else RED
    print(style(f"\n{passed} passed, {failed} failed, {len(RESULTS)} total\n", BOLD, color))
    raise SystemExit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
