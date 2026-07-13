"""Text loading, tokenization, chunking, and scoring helpers for document RAG."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+")
BOUNDARY_RE = re.compile(r"(\n\n+|(?<=[.!?])\s+)")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def tokenize(text: str) -> list[str]:
    """Tokenize for document search.

    Keeps full code-ish tokens like ``tests.py`` and also emits useful subparts
    like ``tests`` and ``py``. That makes exact command/file retrieval work
    without losing natural-language search.
    """
    tokens: list[str] = []
    for raw in TOKEN_RE.findall(text):
        token = raw.lower()
        tokens.append(token)
        tokens.extend(part for part in re.split(r"[./:-]+", token) if part)
    return tokens


def query_terms(query: str) -> list[str]:
    return [term for term in tokenize(query) if term not in STOPWORDS and len(term) > 1]


def normalize_text(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF upload requires pypdf. Install it with: pip install pypdf") from exc

        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            pages.append(f"\n\n[page {i}]\n{page.extract_text() or ''}")
        return "".join(pages)

    return path.read_text(encoding="utf-8", errors="replace")


def split_chunks(
    text: str,
    *,
    chunk_size: int = 1100,
    chunk_overlap: int = 180,
    min_chunk_chars: int = 120,
) -> list[tuple[str, int, int]]:
    text = normalize_text(text)
    if not text:
        return []

    chunks: list[tuple[str, int, int]] = []
    start = 0
    while start < len(text):
        hard_end = min(start + chunk_size, len(text))
        window = text[start:hard_end]
        end = hard_end
        if hard_end < len(text):
            matches = list(BOUNDARY_RE.finditer(window))
            if matches:
                candidate = start + matches[-1].end()
                if candidate - start >= min_chunk_chars:
                    end = candidate
        chunk = text[start:end].strip()
        if len(chunk) >= min_chunk_chars:
            chunks.append((chunk, start, end))
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def cosine_counter(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    shared = set(left) & set(right)
    dot = sum(left[token] * right[token] for token in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)
