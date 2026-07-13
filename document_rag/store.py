"""File-backed document upload and retrieval store.

This is deliberately document-scoped. It stores chunk metadata and text in a
human-readable JSON file under ``document_rag/library``. Retrieval is hybrid:
BM25-style lexical scoring + lightweight semantic similarity + phrase bonuses,
then MMR diversity so one long document does not flood every result.
"""

from __future__ import annotations

import json
import math
import shutil
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from document_rag.text import cosine_counter, load_text, query_terms, split_chunks, tokenize

PACKAGE_DIR = Path(__file__).resolve().parent
LIBRARY_DIR = PACKAGE_DIR / "library"
DEFAULT_INDEX_PATH = LIBRARY_DIR / "documents.json"
ALLOWED_EXTENSIONS = {".txt", ".md", ".rst", ".py", ".json", ".yaml", ".yml", ".pdf"}


@dataclass
class SearchHit:
    chunk: dict[str, Any]
    score: float
    lexical_score: float
    semantic_score: float
    phrase_score: float
    matched_terms: list[str]

    def citation(self) -> str:
        doc_title = self.chunk.get("title") or self.chunk.get("source_name") or self.chunk["doc_id"]
        return f"{doc_title}#chunk-{self.chunk['chunk_index']}"


class DocumentRagStore:
    def __init__(self, index_path: Path | str = DEFAULT_INDEX_PATH):
        self.index_path = Path(index_path)
        self.library_dir = self.index_path.parent
        self.upload_dir = self.library_dir / "uploads"

    def _empty(self) -> dict[str, Any]:
        return {"version": 1, "documents": [], "chunks": []}

    def _load(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return self._empty()
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return self._empty()
        if not isinstance(data, dict):
            return self._empty()
        data.setdefault("version", 1)
        data.setdefault("documents", [])
        data.setdefault("chunks", [])
        return data

    def _save(self, data: dict[str, Any]) -> None:
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def upload(
        self,
        path: Path | str,
        *,
        title: str | None = None,
        copy_file: bool = True,
        chunk_size: int = 1100,
        chunk_overlap: int = 180,
    ) -> dict[str, Any]:
        source = Path(path).expanduser()
        if not source.exists():
            raise FileNotFoundError(f"Document not found: {source}")
        if source.suffix.lower() not in ALLOWED_EXTENSIONS:
            allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
            raise ValueError(f"Unsupported document type {source.suffix!r}. Allowed: {allowed}")

        raw_text = load_text(source)
        chunks = split_chunks(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            raise ValueError("No readable text chunks found in that document.")

        data = self._load()
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source.resolve()}:{source.stat().st_mtime_ns}"))
        doc_title = title or source.stem
        stored_path = str(source)
        if copy_file:
            self.upload_dir.mkdir(parents=True, exist_ok=True)
            destination = self.upload_dir / f"{doc_id}{source.suffix.lower()}"
            shutil.copy2(source, destination)
            stored_path = str(destination)

        # Re-uploading the same file/version replaces the previous chunks.
        data["documents"] = [doc for doc in data["documents"] if doc["doc_id"] != doc_id]
        data["chunks"] = [chunk for chunk in data["chunks"] if chunk["doc_id"] != doc_id]

        document = {
            "doc_id": doc_id,
            "title": doc_title,
            "source_path": str(source),
            "stored_path": stored_path,
            "source_name": source.name,
            "extension": source.suffix.lower(),
            "uploaded": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "chunk_count": len(chunks),
        }
        data["documents"].append(document)

        for i, (text, start, end) in enumerate(chunks):
            data["chunks"].append(
                {
                    "chunk_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{i}:{text}")),
                    "doc_id": doc_id,
                    "title": doc_title,
                    "source_name": source.name,
                    "chunk_index": i,
                    "start_char": start,
                    "end_char": end,
                    "text": text,
                }
            )

        self._save(data)
        return document

    def documents(self) -> list[dict[str, Any]]:
        return self._load()["documents"]

    def clear(self) -> None:
        self._save(self._empty())

    def search(self, query: str, *, top_k: int = 5, candidate_k: int = 24) -> list[SearchHit]:
        data = self._load()
        chunks = data["chunks"]
        if not chunks:
            return []

        q_terms = query_terms(query)
        q_counter = Counter(q_terms)
        if not q_terms:
            q_terms = tokenize(query)
            q_counter = Counter(q_terms)

        chunk_tokens = [tokenize(chunk["text"]) for chunk in chunks]
        chunk_counters = [Counter(tokens) for tokens in chunk_tokens]
        avg_len = sum(len(tokens) for tokens in chunk_tokens) / max(1, len(chunk_tokens))
        doc_freq = Counter(token for tokens in chunk_tokens for token in set(tokens))

        scored: list[SearchHit] = []
        for chunk, tokens, counter in zip(chunks, chunk_tokens, chunk_counters):
            lexical = self._bm25(q_terms, counter, len(tokens), avg_len, doc_freq, len(chunks))
            semantic = cosine_counter(q_counter, counter)
            phrase = self._phrase_score(query, chunk["text"])
            matched = sorted(set(q_terms) & set(tokens))
            score = (0.58 * lexical) + (0.30 * semantic) + (0.12 * phrase)
            if score > 0:
                scored.append(SearchHit(chunk, score, lexical, semantic, phrase, matched))

        scored.sort(key=lambda hit: hit.score, reverse=True)
        return self._mmr(scored[:candidate_k], top_k=top_k)

    def _bm25(
        self,
        terms: list[str],
        counter: Counter[str],
        length: int,
        avg_len: float,
        doc_freq: Counter[str],
        total_docs: int,
    ) -> float:
        if not terms or not counter:
            return 0.0
        k1 = 1.5
        b = 0.75
        score = 0.0
        for term in terms:
            freq = counter.get(term, 0)
            if not freq:
                continue
            idf = math.log(1 + (total_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denom = freq + k1 * (1 - b + b * length / max(avg_len, 1))
            score += idf * ((freq * (k1 + 1)) / denom)
        return score

    def _phrase_score(self, query: str, text: str) -> float:
        query = " ".join(query.lower().split())
        text = " ".join(text.lower().split())
        if not query or not text:
            return 0.0
        if query in text:
            return 1.0
        terms = query_terms(query)
        if len(terms) < 2:
            return 0.0
        bigrams = [" ".join(pair) for pair in zip(terms, terms[1:])]
        return sum(1 for bigram in bigrams if bigram in text) / max(1, len(bigrams))

    def _mmr(self, candidates: list[SearchHit], *, top_k: int) -> list[SearchHit]:
        selected: list[SearchHit] = []
        remaining = candidates[:]
        while remaining and len(selected) < top_k:
            if not selected:
                selected.append(remaining.pop(0))
                continue
            best_idx = 0
            best_score = float("-inf")
            for i, hit in enumerate(remaining):
                diversity_penalty = max(
                    cosine_counter(Counter(tokenize(hit.chunk["text"])), Counter(tokenize(prev.chunk["text"])))
                    for prev in selected
                )
                mmr_score = (0.75 * hit.score) - (0.25 * diversity_penalty)
                if mmr_score > best_score:
                    best_idx = i
                    best_score = mmr_score
            selected.append(remaining.pop(best_idx))
        return selected


_active = DocumentRagStore()


def get_document_store() -> DocumentRagStore:
    return _active


def set_document_store(store: DocumentRagStore) -> None:
    global _active
    _active = store
