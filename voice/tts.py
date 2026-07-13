"""TTS layer — Deepgram Aura streaming synthesis. Ported from Karn's plain/tts.py.

speak_sentences() consumes an async iterator of sentence-sized text chunks
and synthesizes each one as it arrives, so playback starts before the LLM
has finished generating the full reply. Designed to run as an asyncio.Task
so it can be cancelled mid-utterance for barge-in.
"""

from __future__ import annotations

import asyncio
import logging
from array import array
from typing import AsyncIterator

import aiohttp
import pyaudio

from voice.config import CHUNK_SIZE, DEEPGRAM_API_KEY, SAMPLE_RATE, TTS_MODEL, TTS_VOLUME

logger = logging.getLogger(__name__)

FORMAT = pyaudio.paInt16
CHANNELS = 1


def _scale_linear16(chunk: bytes, volume: float) -> bytes:
    if volume == 1.0 or not chunk:
        return chunk
    samples = array("h")
    samples.frombytes(chunk)
    if samples.itemsize != 2:
        return chunk
    for index, sample in enumerate(samples):
        scaled = int(sample * volume)
        samples[index] = max(-32768, min(32767, scaled))
    return samples.tobytes()


async def _speak_one(session: aiohttp.ClientSession, text: str, stream) -> None:
    url = (
        f"https://api.deepgram.com/v1/speak"
        f"?model={TTS_MODEL}&encoding=linear16&sample_rate={SAMPLE_RATE}"
    )
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/json",
    }
    async with session.post(url, headers=headers, json={"text": text}) as resp:
        if resp.status != 200:
            body = await resp.text()
            logger.error("TTS HTTP %s: %s", resp.status, body[:200])
            return
        async for chunk in resp.content.iter_chunked(CHUNK_SIZE):
            if chunk:
                await asyncio.to_thread(stream.write, _scale_linear16(chunk, TTS_VOLUME))


async def speak_sentences(sentences: AsyncIterator[str], audio_out: pyaudio.PyAudio) -> None:
    """Synthesize and play each sentence as it arrives.

    Cancel the wrapping asyncio.Task to implement barge-in — playback stops
    cleanly and the audio device is released.
    """
    stream = audio_out.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, output=True)
    try:
        async with aiohttp.ClientSession() as session:
            async for sentence in sentences:
                try:
                    await _speak_one(session, sentence, stream)
                except OSError:
                    break
    except asyncio.CancelledError:
        pass  # barge-in — clean exit
    except Exception:
        logger.exception("TTS error")
    finally:
        try:
            stream.stop_stream()
            stream.close()
        except OSError:
            pass
