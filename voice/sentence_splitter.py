"""Buffers streamed text deltas and yields complete sentences.

Lets TTS start speaking sentence 1 while the LLM is still generating
sentence 3, instead of waiting for the full reply before synthesizing
anything.
"""

from __future__ import annotations

SENTENCE_ENDERS = (".", "!", "?")


class SentenceSplitter:
    def __init__(self) -> None:
        self._buf = ""

    def feed(self, delta: str) -> list[str]:
        """Add a text delta, return any complete sentences it completed."""
        self._buf += delta
        sentences = []
        while True:
            cut = self._find_break(self._buf)
            if cut is None:
                break
            sentence, self._buf = self._buf[:cut].strip(), self._buf[cut:]
            if sentence:
                sentences.append(sentence)
        return sentences

    def flush(self) -> str | None:
        """Call once the stream ends; returns any trailing partial sentence."""
        remainder = self._buf.strip()
        self._buf = ""
        return remainder or None

    @staticmethod
    def _find_break(text: str) -> int | None:
        # A newline always ends a sentence. Punctuation only counts when the
        # NEXT character is whitespace — this keeps decimals ("3.14") and
        # dotted names intact, at the cost of holding a genuinely final
        # sentence in the buffer until flush() (which the stream end calls).
        for i, ch in enumerate(text):
            if ch == "\n":
                return i + 1
            if ch in SENTENCE_ENDERS and i + 1 < len(text) and text[i + 1].isspace():
                return i + 1
        return None
