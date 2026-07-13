"""Deepgram Flux streaming STT. Ported from Karn's plain/stt.py.

Flux streams partial transcripts and emits turn events (StartOfTurn, Update,
EagerEndOfTurn, EndOfTurn) so the session can react to barge-in and finalized
turns without waiting on silence-based VAD.
"""

from __future__ import annotations

import urllib.parse

import websockets

from voice.config import DEEPGRAM_API_KEY, SAMPLE_RATE, STT_MODEL


def build_flux_url() -> str:
    params = {
        "model": STT_MODEL,
        "encoding": "linear16",
        "sample_rate": str(SAMPLE_RATE),
        "eot_threshold": "0.7",
        "eot_timeout_ms": "800",
        "eager_eot_threshold": "0.7",
    }
    return f"wss://api.deepgram.com/v2/listen?{urllib.parse.urlencode(params)}"


def connect():
    """Open a Flux WebSocket connection."""
    return websockets.connect(
        build_flux_url(),
        additional_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
    )
