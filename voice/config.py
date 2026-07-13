"""Voice pipeline configuration. Reads from the same .env as the rest of Caranthir."""

import os

DEEPGRAM_API_KEY = (os.getenv("DEEPGRAM_API_KEY") or "").strip()

SAMPLE_RATE = 16_000
CHANNELS = 1
CHUNK_SIZE = 1_280

STT_MODEL = os.getenv("VOICE_STT_MODEL", "flux-general-en")
TTS_MODEL = os.getenv("VOICE_TTS_MODEL", "aura-2-thalia-en")
TTS_VOLUME = max(0.0, min(float(os.getenv("VOICE_TTS_VOLUME", "1.0")), 2.0))


def require_keys() -> None:
    if not DEEPGRAM_API_KEY or len(DEEPGRAM_API_KEY) < 32:
        raise SystemExit("Missing or invalid DEEPGRAM_API_KEY in .env — required for voice mode.")


def require_native_audio() -> None:
    """Bail out early if running inside WSL, where native audio is unavailable."""
    if os.environ.get("WSL_DISTRO_NAME"):
        _wsl_exit()
    try:
        with open("/proc/version", encoding="utf-8") as f:
            if "microsoft" in f.read().lower():
                _wsl_exit()
    except OSError:
        pass


def _wsl_exit() -> None:
    raise SystemExit(
        "\nWSL cannot access your Windows microphone/speakers.\n"
        "Run Caranthir from PowerShell instead for voice mode.\n"
    )
