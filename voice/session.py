"""Voice session loop (mic + speakers), modeled on Karn's plain/agent.py.

Pipeline:
  mic -> Deepgram Flux STT -> LangGraph agent (astream) -> sentence splitter
       -> Deepgram Aura TTS -> speakers

Reuses the same LangGraph agent, checkpointer, and thread_id as the text
REPL, so conversation history is shared across text and voice turns. Runs
under its own asyncio event loop, entered and exited around the normal sync
REPL loop in main.py.
"""

from __future__ import annotations

import asyncio
import json

import pyaudio
from langchain_core.messages import AIMessageChunk, HumanMessage

from memory import open_async_checkpointer
from voice import config, stt, tts
from voice.sentence_splitter import SentenceSplitter

DIM = "\033[2m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _default_input_device(pa: pyaudio.PyAudio) -> int:
    try:
        return int(pa.get_default_input_device_info()["index"])
    except OSError as exc:
        raise SystemExit(f"No microphone found: {exc}") from exc


async def _astream_reply_sentences(agent, prompt: str, config_dict: dict):
    """Run one agent turn, yielding complete sentences as text streams in.

    Tool-call activity is printed inline but not spoken; only final AI text
    is sent to TTS.
    """
    splitter = SentenceSplitter()
    async for chunk, _metadata in agent.astream(
        {"messages": [HumanMessage(content=prompt)]},
        config=config_dict,
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessageChunk):
            text = chunk.content if isinstance(chunk.content, str) else _extract_text(chunk.content)
            if text:
                for sentence in splitter.feed(text):
                    yield sentence
        elif hasattr(chunk, "name") and hasattr(chunk, "content") and not isinstance(chunk, AIMessageChunk):
            # ToolMessage: tool already ran; nothing to speak, just log.
            print(f"{DIM}  * tool {getattr(chunk, 'name', '?')}{RESET}")
    trailing = splitter.flush()
    if trailing:
        yield trailing


def _extract_text(content: list) -> str:
    parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "".join(parts)


async def run_session(agent, thread_config: dict) -> None:
    """Run an interactive voice session until the user asks to stop.

    `agent` should already be built with the voice-restricted tool set.
    `thread_config` is the same LangGraph config dict used by the text REPL
    (same thread_id), so history carries over between modes.
    """
    config.require_keys()
    config.require_native_audio()

    audio = pyaudio.PyAudio()
    input_device = _default_input_device(audio)

    agent_speaking = False
    speak_task: asyncio.Task | None = None

    print(f"{CYAN}Voice mode: connecting to Deepgram...{RESET}")

    async with stt.connect() as ws:
        mic = audio.open(
            format=pyaudio.paInt16,
            channels=config.CHANNELS,
            rate=config.SAMPLE_RATE,
            input=True,
            input_device_index=input_device,
            frames_per_buffer=config.CHUNK_SIZE,
        )

        stop_requested = asyncio.Event()

        async def send_audio() -> None:
            try:
                while True:
                    chunk = await asyncio.to_thread(mic.read, config.CHUNK_SIZE, False)
                    if not agent_speaking:
                        await ws.send(chunk)
            except asyncio.CancelledError:
                pass

        async def say(sentences) -> None:
            nonlocal agent_speaking, speak_task
            agent_speaking = True
            speak_task = asyncio.create_task(tts.speak_sentences(sentences, audio))

            def _done(t: asyncio.Task) -> None:
                nonlocal agent_speaking
                if not t.cancelled():
                    agent_speaking = False

            speak_task.add_done_callback(_done)
            await speak_task

        async def handle_messages() -> None:
            nonlocal agent_speaking, speak_task
            try:
                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    event = json.loads(raw)
                    event_type = event.get("type", "")

                    if event_type in ("ListenV2Connected", "Connected"):
                        print(f"{DIM}[voice: connected]{RESET}")

                    elif event_type in ("ListenV2TurnInfo", "TurnInfo"):
                        turn_event = event.get("event", "")
                        transcript = event.get("transcript", "").strip()

                        if turn_event == "StartOfTurn":
                            if agent_speaking and speak_task and not speak_task.done():
                                speak_task.cancel()
                                agent_speaking = False
                                print(f"{YELLOW}[barge-in]{RESET}")

                        elif turn_event == "EndOfTurn" and transcript:
                            print(f"\n{BOLD}You:{RESET} {transcript}")

                            if transcript.lower().strip(" .!") in (
                                "stop talking",
                                "exit voice mode",
                                "stop voice mode",
                            ):
                                stop_requested.set()
                                return

                            printed = {"any": False}

                            async def _sentences():
                                # Agent/LLM errors are caught HERE, not left to
                                # propagate into tts.speak_sentences' broad
                                # except — otherwise they get logged as "TTS
                                # error" and point debugging at the wrong layer.
                                try:
                                    async for sentence in _astream_reply_sentences(
                                        agent, transcript, thread_config
                                    ):
                                        if not printed["any"]:
                                            print(f"{GREEN}{BOLD}Caranthir:{RESET} ", end="")
                                            printed["any"] = True
                                        print(sentence, end=" ", flush=True)
                                        yield sentence
                                    print()
                                except asyncio.CancelledError:
                                    raise
                                except Exception as exc:
                                    print(f"\n{YELLOW}[agent error — turn abandoned] {exc}{RESET}")

                            await say(_sentences())

                    elif event_type in ("ListenV2FatalError", "FatalError"):
                        print(f"{DIM}[voice error] {event}{RESET}")
                        return
            except asyncio.CancelledError:
                pass

        send_task = asyncio.create_task(send_audio())
        recv_task = asyncio.create_task(handle_messages())

        print(f"{DIM}[mic live] Speak, or say \"stop talking\" to return to text. Ctrl+C to quit.{RESET}\n")

        stop_wait = asyncio.create_task(stop_requested.wait())
        try:
            await asyncio.wait(
                [send_task, recv_task, stop_wait],
                return_when=asyncio.FIRST_COMPLETED,
            )
        except KeyboardInterrupt:
            pass
        finally:
            for task in (send_task, recv_task, stop_wait):
                task.cancel()
            if speak_task and not speak_task.done():
                speak_task.cancel()
            await asyncio.gather(send_task, recv_task, stop_wait, return_exceptions=True)
            mic.stop_stream()
            mic.close()
            audio.terminate()
            print(f"\n{DIM}[voice mode ended]{RESET}\n")


async def run_voice_mode(build_voice_agent, model: str, provider: str | None, thread_config: dict) -> None:
    """Entry point called from main.py's sync REPL loop.

    Opens the async-compatible checkpointer (agent.astream() requires one;
    the text REPL's SqliteSaver is sync-only) against the same on-disk
    checkpoints.sqlite, so voice and text turns share history via thread_config's
    thread_id. `build_voice_agent` is main.build_voice_agent, passed in to avoid
    a main -> voice -> main import cycle.
    """
    async with open_async_checkpointer() as async_checkpointer:
        agent, _ = build_voice_agent(model, provider, checkpointer=async_checkpointer)
        await run_session(agent, thread_config)
