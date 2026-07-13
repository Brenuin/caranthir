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

from main import (
    build_agent,
    build_llm,
    extract_text,
    extract_thinking,
    infer_provider,
    new_ai_text,
)
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


def test_extract_thinking():
    content = [
        {"type": "thinking", "thinking": "step one", "signature": "s"},
        {"type": "text", "text": "the answer"},
        {"type": "thinking", "thinking": "", "signature": "s2"},  # omitted-display block
    ]
    assert extract_thinking(content) == "step one"
    assert extract_thinking("plain string") == ""


def test_show_thinking_guard_openai():
    try:
        build_llm("gpt-4.1-mini", None, None, show_thinking=True)
        raise AssertionError("expected ValueError for --show-thinking on OpenAI")
    except ValueError as exc:
        assert "thinking" in str(exc).lower()


def test_memory_store_roundtrip():
    import os
    import tempfile

    from memory import MemoryStore

    path = os.path.join(tempfile.mkdtemp(), "memories.json")
    store = MemoryStore(path)
    assert store.all() == []

    first = store.add("Master's cat is named Beans")
    assert first["id"] == 1
    # Case-insensitive dedupe: saving the same fact again is a no-op.
    duplicate = store.add("master's cat is named beans")
    assert duplicate["id"] == 1 and len(store.all()) == 1

    second = store.add("Prefers dark roast coffee")
    assert second["id"] == 2

    # Persists across store instances (i.e. across sessions).
    fresh = MemoryStore(path)
    assert [e["text"] for e in fresh.all()] == [
        "Master's cat is named Beans",
        "Prefers dark roast coffee",
    ]
    assert fresh.forget(1) is True
    assert fresh.forget(99) is False
    assert [e["id"] for e in fresh.all()] == [2]


def test_memory_context_block():
    import os
    import tempfile

    from memory import MemoryStore, get_store, memory_context, set_store

    original = get_store()
    try:
        set_store(MemoryStore(os.path.join(tempfile.mkdtemp(), "memories.json")))
        assert memory_context() == ""
        get_store().add("Likes vermilion")
        block = memory_context()
        assert "[1] Likes vermilion" in block
        assert "Long-term memory" in block
    finally:
        set_store(original)


def test_document_rag_upload_and_search():
    import tempfile
    from pathlib import Path

    from document_rag import DocumentRagStore

    tmp = Path(tempfile.mkdtemp())
    source = tmp / "agent_notes.md"
    source.write_text(
        """
        # Agent notes

        Hybrid document retrieval should combine exact lexical matches with
        semantic matching. Commands and filenames need lexical search because
        embeddings can blur exact identifiers.

        To test Caranthir safely, run python3 tests.py --skip-live.
        Live API tests should only run when explicitly requested.
        """,
        encoding="utf-8",
    )

    store = DocumentRagStore(tmp / "library" / "documents.json")
    document = store.upload(source, title="Agent Notes")
    assert document["title"] == "Agent Notes"
    assert document["chunk_count"] >= 1

    hits = store.search("What command tests Caranthir safely?", top_k=3)
    assert hits, "expected document search hits"
    best_text = hits[0].chunk["text"].lower()
    assert "python3 tests.py --skip-live" in best_text
    assert "tests" in hits[0].matched_terms


def test_document_rag_store_is_separate_from_memory_store():
    import tempfile
    from pathlib import Path

    from document_rag import DocumentRagStore
    from memory import MemoryStore

    tmp = Path(tempfile.mkdtemp())
    doc_store = DocumentRagStore(tmp / "docs" / "documents.json")
    memory_store = MemoryStore(tmp / "memories.json")

    memory_store.add("User prefers concise answers")
    assert memory_store.all()
    assert doc_store.documents() == []


def test_sqlite_checkpointer_persists():
    """LangGraph's SqliteSaver must restore a thread's messages from disk
    into a brand-new graph — no API needed, the node is a stub."""
    import os
    import tempfile

    from langgraph.graph import START, MessagesState, StateGraph

    from memory import open_checkpointer

    path = os.path.join(tempfile.mkdtemp(), "checkpoints.sqlite")

    def echo(state: MessagesState):
        return {"messages": [AIMessage(content="noted")]}

    def build(saver):
        graph = StateGraph(MessagesState)
        graph.add_node("echo", echo)
        graph.add_edge(START, "echo")
        return graph.compile(checkpointer=saver)

    config = {"configurable": {"thread_id": "t"}}
    graph = build(open_checkpointer(path))
    graph.invoke({"messages": [HumanMessage(content="my number is 47")]}, config)

    # Simulate a restart: new saver instance, same file, fresh graph.
    graph2 = build(open_checkpointer(path))
    restored = [m.content for m in graph2.get_state(config).values["messages"]]
    assert restored == ["my number is 47", "noted"], f"unexpected state: {restored}"


def test_agent_data_dir_namespaces():
    """Namespaced agents get their own subdirectory, isolated from
    Caranthir's own top-level checkpoints.sqlite / memories.json."""
    import shutil
    import tempfile
    from pathlib import Path

    import memory.checkpoints as checkpoints_module

    tmp_root = Path(tempfile.mkdtemp())
    original = checkpoints_module.AGENTS_DIR
    checkpoints_module.AGENTS_DIR = tmp_root / "agents"
    try:
        d1 = checkpoints_module.agent_data_dir("task_agent")
        d2 = checkpoints_module.agent_data_dir("task_agent")
        assert d1 == d2, "same namespace must resolve to the same directory"
        assert d1.exists()

        other = checkpoints_module.agent_data_dir("another_agent")
        assert other != d1, "different namespaces must not collide"
    finally:
        checkpoints_module.AGENTS_DIR = original
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_hotreload_picks_up_changes():
    import os

    import prompts
    from hotreload import check_and_reload

    # First calls only record file mtimes; nothing has changed yet.
    assert check_and_reload() is None
    assert check_and_reload() is None
    # Simulate an edit by bumping prompts.py's mtime.
    stat = os.stat(prompts.__file__)
    os.utime(prompts.__file__, (stat.st_atime, stat.st_mtime + 1))
    fresh = check_and_reload()
    assert fresh is not None, "expected a reload after the file changed"
    assert callable(fresh.build_agent)
    # And back to quiet once the change has been absorbed.
    assert check_and_reload() is None


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


# ------------------------------------------------- voice + task-agent units


class ScriptedChatModel:
    """Minimal stand-in for a chat model: returns scripted AIMessages in
    order, repeating the last one. bind_tools is a no-op so the task-agent
    graph can be exercised with zero API calls."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, **kwargs):
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


def test_sentence_splitter():
    from voice.sentence_splitter import SentenceSplitter

    # Boundaries only where punctuation is followed by whitespace.
    s = SentenceSplitter()
    out = []
    for delta in ["Hello", " there", ". How", " are you", " today? ", "Great!"]:
        out.extend(s.feed(delta))
    trailing = s.flush()
    if trailing:
        out.append(trailing)
    assert out == ["Hello there.", "How are you today?", "Great!"], f"got: {out}"

    # Decimals and dotted tokens must not be split mid-number.
    s = SentenceSplitter()
    out = []
    for delta in ["Pi is 3", ".14159 and", " that is neat", ". Version 2.5 shipped."]:
        out.extend(s.feed(delta))
    trailing = s.flush()
    if trailing:
        out.append(trailing)
    assert out == ["Pi is 3.14159 and that is neat.", "Version 2.5 shipped."], f"got: {out}"

    # Newlines always break, even with no trailing space.
    s = SentenceSplitter()
    out = s.feed("line one\nline two\n")
    assert out == ["line one", "line two"], f"got: {out}"

    # flush() is a one-shot drain.
    s = SentenceSplitter()
    s.feed("partial sentence")
    assert s.flush() == "partial sentence"
    assert s.flush() is None


def test_voice_config_key_guard():
    from voice import config as voice_config

    original = voice_config.DEEPGRAM_API_KEY
    try:
        for bad in ("", "too-short"):
            voice_config.DEEPGRAM_API_KEY = bad
            try:
                voice_config.require_keys()
                raise AssertionError(f"expected SystemExit for key {bad!r}")
            except SystemExit:
                pass
        # A plausible key length passes.
        voice_config.DEEPGRAM_API_KEY = "x" * 40
        voice_config.require_keys()
    finally:
        voice_config.DEEPGRAM_API_KEY = original


def test_flux_url_params():
    from voice import stt
    from voice.config import SAMPLE_RATE, STT_MODEL

    url = stt.build_flux_url()
    assert url.startswith("wss://api.deepgram.com/v2/listen?")
    assert f"model={STT_MODEL}" in url
    assert f"sample_rate={SAMPLE_RATE}" in url
    assert "encoding=linear16" in url
    assert "eot_threshold=" in url and "eot_timeout_ms=" in url


def test_tts_volume_scaling():
    from array import array

    from voice.tts import _scale_linear16

    samples = array("h", [1000, -1000, 30000, -30000]).tobytes()
    # volume 1.0 is a passthrough (same object, no copy).
    assert _scale_linear16(samples, 1.0) is samples
    assert _scale_linear16(b"", 0.5) == b""

    halved = array("h")
    halved.frombytes(_scale_linear16(samples, 0.5))
    assert list(halved) == [500, -500, 15000, -15000]

    # Doubling must clip at int16 bounds instead of overflowing.
    doubled = array("h")
    doubled.frombytes(_scale_linear16(samples, 2.0))
    assert list(doubled) == [2000, -2000, 32767, -32768]


def test_stream_turn_voice_sentinel():
    from main import VOICE_MODE_SENTINEL, stream_turn
    from terminal_ui import StreamPrinter

    class FakeAgent:
        def __init__(self, chunks):
            self.chunks = chunks

        def stream(self, _input, config=None, stream_mode=None):
            yield from self.chunks

    printer = StreamPrinter("Caranthir")
    # A turn where start_voice fires must return True.
    agent = FakeAgent([
        (AIMessage(content="Switching to voice."), {}),
        (ToolMessage(content=VOICE_MODE_SENTINEL, name="start_voice", tool_call_id="c1"), {}),
    ])
    assert stream_turn(agent, "let's talk", {}, printer) is True
    printer.finish()

    # An ordinary tool call must not trip the switch.
    printer = StreamPrinter("Caranthir")
    agent = FakeAgent([
        (ToolMessage(content="2026-07-12", name="get_current_time", tool_call_id="c2"), {}),
        (AIMessage(content="It is July twelfth."), {}),
    ])
    assert stream_turn(agent, "what time", {}, printer) is False
    printer.finish()


def test_voice_toolset_excludes_start_voice():
    from main import TOOLS, VOICE_TOOLS

    names = {t.name for t in TOOLS}
    voice_names = {t.name for t in VOICE_TOOLS}
    assert "start_voice" in names, "text-mode agent must offer start_voice"
    assert "start_voice" not in voice_names, "voice agent must not re-trigger voice mode"
    # Voice keeps the basics.
    assert {"get_current_time", "remember", "forget"} <= voice_names


def test_astream_reply_sentences():
    import asyncio

    from langchain_core.messages import AIMessageChunk

    from voice.session import _astream_reply_sentences

    class FakeAsyncAgent:
        def __init__(self, chunks):
            self.chunks = chunks

        async def astream(self, _input, config=None, stream_mode=None):
            for chunk in self.chunks:
                yield chunk

    agent = FakeAsyncAgent([
        (AIMessageChunk(content="Hello there. How"), {}),
        (AIMessageChunk(content=" are you today?"), {}),
        (ToolMessage(content="ignored", name="get_current_time", tool_call_id="c3"), {}),
        (AIMessageChunk(content=" All good."), {}),
    ])

    async def collect():
        return [s async for s in _astream_reply_sentences(agent, "hi", {})]

    sentences = asyncio.run(collect())
    assert sentences == ["Hello there.", "How are you today?", "All good."], f"got: {sentences}"


def test_task_channel():
    from tasks import SharedTaskChannel

    ch = SharedTaskChannel()
    ch.publish("t1", "progress", "step one")
    ch.publish("t1", "progress", "step two")
    ch.publish("t2", "progress", "other task")

    unread = ch.unread("t1")
    assert [(e.kind, e.text) for e in unread] == [
        ("progress", "step one"),
        ("progress", "step two"),
    ]
    # Reading t1 must not consume t2's entries.
    ch.mark_read(unread)
    assert ch.unread("t1") == []
    assert len(ch.unread("t2")) == 1

    ch.publish("t1", "report", "all done")
    fresh = ch.unread("t1")
    assert len(fresh) == 1 and fresh[0].kind == "report"

    assert ch.last_activity_at("t1") is not None
    assert ch.last_activity_at("nope") is None


def test_task_agent_graph_offline():
    """Full task-agent graph pass with a scripted model: a tool-calling step
    must publish a progress entry, and run_task must publish the report."""
    from langgraph.checkpoint.memory import MemorySaver

    from langchain_core.tools import tool as tool_decorator

    from tasks import SharedTaskChannel, build_task_agent, run_task

    @tool_decorator
    def add_numbers(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    llm = ScriptedChatModel([
        AIMessage(
            content="",
            tool_calls=[{"name": "add_numbers", "args": {"a": 17, "b": 25}, "id": "c1", "type": "tool_call"}],
        ),
        AIMessage(content="The result is 42."),
    ])
    channel = SharedTaskChannel()
    agent = build_task_agent(
        llm, [add_numbers], task_id="offline-1", channel=channel, checkpointer=MemorySaver()
    )
    report = run_task(agent, "offline-1", "Add 17 and 25.", channel=channel)

    assert report == "The result is 42."
    kinds = [(e.kind, e.text) for e in channel.all_for("offline-1")]
    assert kinds == [
        ("progress", "working: add_numbers"),
        ("report", "The result is 42."),
    ], f"got: {kinds}"


def test_task_agent_persists_state():
    """Two agent instances sharing a checkpointer file must accumulate one
    thread's history — the 'revive a dormant task-agent' guarantee."""
    import os
    import tempfile

    from langchain_core.tools import tool as tool_decorator

    from memory import open_checkpointer
    from tasks import SharedTaskChannel, build_task_agent, run_task

    @tool_decorator
    def noop() -> str:
        """Placeholder tool."""
        return "ok"

    path = os.path.join(tempfile.mkdtemp(), "checkpoints.sqlite")
    channel = SharedTaskChannel()
    config = {"configurable": {"thread_id": "persist-1"}}

    agent1 = build_task_agent(
        ScriptedChatModel([AIMessage(content="Noted.")]),
        [noop], task_id="persist-1", channel=channel,
        checkpointer=open_checkpointer(path),
    )
    run_task(agent1, "persist-1", "The target file is report_q3.csv.", channel=channel)

    agent2 = build_task_agent(
        ScriptedChatModel([AIMessage(content="It was report_q3.csv.")]),
        [noop], task_id="persist-1", channel=channel,
        checkpointer=open_checkpointer(path),
    )
    run_task(agent2, "persist-1", "What was the file?", channel=channel)

    restored = [m.content for m in agent2.get_state(config).values["messages"]]
    assert "The target file is report_q3.csv." in restored, f"history lost: {restored}"
    assert len(restored) == 4, f"expected 2 turns (4 messages), got {len(restored)}"


def test_run_task_publishes_error():
    from tasks import SharedTaskChannel
    from tasks.agent import run_task

    class BrokenAgent:
        def invoke(self, *args, **kwargs):
            raise RuntimeError("provider exploded")

    channel = SharedTaskChannel()
    report = run_task(BrokenAgent(), "err-1", "do something", channel=channel)
    assert report.startswith("Task failed:"), f"got: {report}"
    entries = channel.all_for("err-1")
    assert len(entries) == 1 and entries[0].kind == "error"
    assert "provider exploded" in entries[0].text


# ---------------------------------------------------------------- live tests


def _one_turn(agent, prompt: str, thread_id: str):
    return agent.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config={"configurable": {"thread_id": thread_id}},
    )


def local_tool_calls(messages) -> list[tuple[str, str]]:
    """(tool name, result) for every local tool executed via ToolNode."""
    return [
        (m.name, str(m.content))
        for m in messages
        if isinstance(m, ToolMessage)
    ]


def hosted_tool_calls(messages) -> list[tuple[str, str]]:
    """(tool name, input) for every Anthropic server-side tool call."""
    return [
        (block.get("name", "?"), str(block.get("input", "")))
        for m in messages
        if isinstance(m, AIMessage) and isinstance(m.content, list)
        for block in m.content
        if isinstance(block, dict) and block.get("type") == "server_tool_use"
    ]


def show_tool_activity(messages) -> None:
    """Print every tool call observed in this test, so you can SEE the usage."""
    for name, result in local_tool_calls(messages):
        print(style(f"      * local  {name} -> {result[:90]}", DIM))
    for name, tool_input in hosted_tool_calls(messages):
        print(style(f"      * hosted {name} -> {tool_input[:90]}", DIM))


def test_live_openai_local_tool():
    agent, provider = build_agent("gpt-4.1-mini", None, None, False)
    assert provider == "openai"
    result = _one_turn(agent, "What time is it? Use your tool.", "t-openai")
    show_tool_activity(result["messages"])
    calls = local_tool_calls(result["messages"])
    assert calls, "expected get_current_time to be called"
    assert calls[0][0] == "get_current_time"
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
    show_tool_activity(result["messages"])
    calls = hosted_tool_calls(result["messages"])
    assert calls, "expected a server_tool_use block from hosted code execution"
    reply = new_ai_text(result["messages"], set())
    assert "1048576" in reply.replace(",", ""), f"expected 1048576 in reply: {reply!r}"


def test_live_hosted_web_search():
    agent, _ = build_agent("claude-sonnet-5", None, None, True)
    result = _one_turn(
        agent,
        "Search the web for the current stable version of Python and tell me what it is.",
        "t-search",
    )
    show_tool_activity(result["messages"])
    names = [name for name, _ in hosted_tool_calls(result["messages"])]
    assert "web_search" in names, f"expected a web_search call, saw: {names}"
    reply = new_ai_text(result["messages"], set())
    assert reply.strip(), "expected a text answer from the search"


def test_live_show_thinking_summary():
    """--show-thinking should return readable summarized thinking blocks."""
    agent, _ = build_agent("claude-sonnet-5", None, "high", False, show_thinking=True)
    result = _one_turn(
        agent,
        "Three boxes: GG, SS, GS coins. I draw a gold coin from a random box. "
        "Probability the other coin in that box is gold?",
        "t-show-thinking",
    )
    thinking = "\n".join(
        extract_thinking(m.content)
        for m in result["messages"]
        if isinstance(m, AIMessage)
    ).strip()
    assert thinking, "expected non-empty summarized thinking text"
    print(style(f"      ~ thinking: {thinking[:90]}", DIM))


def test_live_memory_across_sessions():
    """A fact saved via the remember tool must survive into a brand-new
    agent with a brand-new thread — that's the point of the memory layer."""
    import os
    import tempfile

    from memory import MemoryStore, get_store, set_store

    original = get_store()
    try:
        set_store(MemoryStore(os.path.join(tempfile.mkdtemp(), "memories.json")))

        agent, _ = build_agent("gpt-4.1-mini", None, None, False)
        result = _one_turn(
            agent, "My cat is named Beans. Save that to your memory.", "t-mem-save"
        )
        show_tool_activity(result["messages"])
        saved = get_store().all()
        assert saved, "expected the remember tool to have written a memory"
        assert "beans" in saved[0]["text"].lower(), f"unexpected memory: {saved!r}"

        # New agent, new thread = a fresh session with no shared history.
        agent2, _ = build_agent("gpt-4.1-mini", None, None, False)
        result = _one_turn(agent2, "What is my cat's name? One word.", "t-mem-recall")
        reply = new_ai_text(result["messages"], set())
        assert "beans" in reply.lower(), f"memory not recalled, reply: {reply!r}"
    finally:
        set_store(original)


def test_live_sqlite_checkpoint_resume():
    """A conversation checkpointed to SQLite must resume in a rebuilt agent —
    same thread id, brand-new agent and saver, shared database file."""
    import os
    import tempfile

    from memory import open_checkpointer

    path = os.path.join(tempfile.mkdtemp(), "checkpoints.sqlite")

    agent, _ = build_agent(
        "gpt-4.1-mini", None, None, False, checkpointer=open_checkpointer(path)
    )
    _one_turn(agent, "My favorite number is 47. Just acknowledge it.", "t-resume")

    agent2, _ = build_agent(
        "gpt-4.1-mini", None, None, False, checkpointer=open_checkpointer(path)
    )
    result = _one_turn(agent2, "What is my favorite number? Just the number.", "t-resume")
    reply = new_ai_text(result["messages"], set())
    assert "47" in reply, f"resumed conversation not recalled, reply: {reply!r}"


def test_live_thinking_model_uses_local_tool():
    """The thinking model (--effort) must still reach local tools through the graph."""
    agent, _ = build_agent("claude-sonnet-5", None, "low", False)
    result = _one_turn(agent, "What time is it right now? Use your tool.", "t-think-tool")
    show_tool_activity(result["messages"])
    calls = local_tool_calls(result["messages"])
    assert calls, "expected the thinking model to call get_current_time"
    assert calls[0][0] == "get_current_time"


def test_live_task_agent_delegation():
    """The task-agent must run a delegated task with a real model: use its
    tool, publish progress while working, and publish the final report."""
    import os
    import tempfile

    from langchain_core.tools import tool as tool_decorator
    from langchain_openai import ChatOpenAI

    from memory import open_checkpointer
    from tasks import SharedTaskChannel, build_task_agent, run_task

    @tool_decorator
    def add_numbers(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    path = os.path.join(tempfile.mkdtemp(), "checkpoints.sqlite")
    channel = SharedTaskChannel()
    agent = build_task_agent(
        ChatOpenAI(model="gpt-4.1-mini"), [add_numbers], task_id="live-task-1",
        channel=channel, checkpointer=open_checkpointer(path),
    )
    report = run_task(
        agent, "live-task-1", "Use your tool to add 17 and 25. State the result plainly.",
        channel=channel,
    )
    print(style(f"      * report: {report[:90]}", DIM))
    assert "42" in report, f"expected 42 in report: {report!r}"

    kinds = [e.kind for e in channel.all_for("live-task-1")]
    assert "progress" in kinds, f"expected a progress entry, got kinds: {kinds}"
    assert kinds[-1] == "report", f"expected the last entry to be the report, got: {kinds}"


# ---------------------------------------------------------------- runner


UNIT_TESTS = [
    ("unit: provider inference", test_infer_provider),
    ("unit: --effort rejected on OpenAI", test_effort_guard_openai),
    ("unit: --hosted-tools rejected on OpenAI", test_hosted_guard_openai),
    ("unit: --show-thinking rejected on OpenAI", test_show_thinking_guard_openai),
    ("unit: extract_text on plain string", test_extract_text_string),
    ("unit: extract_text filters thinking blocks", test_extract_text_blocks),
    ("unit: extract_thinking pulls thinking blocks", test_extract_thinking),
    ("unit: new_ai_text collects and dedupes", test_new_ai_text_collects_and_dedupes),
    ("unit: memory store roundtrip", test_memory_store_roundtrip),
    ("unit: memory context block", test_memory_context_block),
    ("unit: document RAG upload and search", test_document_rag_upload_and_search),
    (
        "unit: document RAG stays separate from memory",
        test_document_rag_store_is_separate_from_memory_store,
    ),
    ("unit: agent_data_dir namespaces isolate agents", test_agent_data_dir_namespaces),
    ("unit: sqlite checkpointer persists to disk", test_sqlite_checkpointer_persists),
    ("unit: hot reload picks up file changes", test_hotreload_picks_up_changes),
    ("unit: sentence splitter boundaries", test_sentence_splitter),
    ("unit: voice config key guard", test_voice_config_key_guard),
    ("unit: Flux STT URL parameters", test_flux_url_params),
    ("unit: TTS volume scaling and clipping", test_tts_volume_scaling),
    ("unit: start_voice sentinel flips stream_turn", test_stream_turn_voice_sentinel),
    ("unit: voice tool set excludes start_voice", test_voice_toolset_excludes_start_voice),
    ("unit: astream reply sentences bridging", test_astream_reply_sentences),
    ("unit: task channel publish/read isolation", test_task_channel),
    ("unit: task-agent graph offline (scripted LLM)", test_task_agent_graph_offline),
    ("unit: task-agent persists across instances", test_task_agent_persists_state),
    ("unit: run_task publishes errors", test_run_task_publishes_error),
]

LIVE_TESTS = [
    ("live: OpenAI local tool loop", test_live_openai_local_tool),
    ("live: Anthropic basic reply", test_live_anthropic_reply),
    ("live: Anthropic memory across turns", test_live_anthropic_memory),
    ("live: memory layer across sessions", test_live_memory_across_sessions),
    ("live: sqlite checkpoint resume", test_live_sqlite_checkpoint_resume),
    ("live: extended thinking (--effort)", test_live_effort_thinking),
    ("live: summarized thinking display (--show-thinking)", test_live_show_thinking_summary),
    ("live: thinking model uses local tool", test_live_thinking_model_uses_local_tool),
    ("live: hosted code execution (--hosted-tools)", test_live_hosted_code_execution),
    ("live: hosted web search (--hosted-tools)", test_live_hosted_web_search),
    ("live: task-agent delegation end to end", test_live_task_agent_delegation),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Caranthir's test suite.")
    parser.add_argument("--skip-live", action="store_true", help="Skip tests that call real APIs.")
    args = parser.parse_args()

    load_dotenv()

    # Point long-term memory at a throwaway file: live tests can make the
    # model call remember(), and that must never touch the real
    # memory/memories.json.
    import os
    import tempfile

    from memory import MemoryStore, set_store

    def isolate_memory() -> None:
        set_store(MemoryStore(os.path.join(tempfile.mkdtemp(), "memories.json")))

    isolate_memory()

    print(style("\n-- unit tests --", BOLD))
    for name, fn in UNIT_TESTS:
        run_test(name, fn)

    if args.skip_live:
        print(style("\n-- live tests skipped (--skip-live) --", DIM))
    else:
        # The hot reload unit test reloads memory/store.py, which resets the
        # active store to the real file — isolate again before going live.
        isolate_memory()
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
