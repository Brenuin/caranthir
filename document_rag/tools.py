"""Agent-facing tools for document upload and retrieval."""

from __future__ import annotations

from langchain_core.tools import tool

from document_rag.store import get_document_store


@tool
def upload_document(path: str, title: str | None = None) -> str:
    """Upload and index a readable document for document-only retrieval.

    Use when the user gives a path to a book, PDF, Markdown, text, code, JSON,
    or YAML document and wants Caranthir to remember/search that document. This
    is for document libraries, not personal long-term memory facts."""
    document = get_document_store().upload(path, title=title)
    return (
        f"Uploaded document '{document['title']}' as {document['doc_id']} "
        f"with {document['chunk_count']} searchable chunks."
    )


@tool
def search_documents(query: str, top_k: int = 5) -> str:
    """Search uploaded documents and return cited evidence chunks.

    Use for questions that should be answered from books/documents the user has
    uploaded. Prefer this over general memory when the user asks about a
    document, book, PDF, manual, notes, or uploaded text."""
    top_k = max(1, min(int(top_k), 10))
    hits = get_document_store().search(query, top_k=top_k)
    if not hits:
        return "No uploaded document evidence matched that query."

    parts = []
    for i, hit in enumerate(hits, start=1):
        snippet = hit.chunk["text"].replace("\n", " ")
        if len(snippet) > 900:
            snippet = snippet[:897] + "..."
        parts.append(
            f"[{i}] {hit.citation()} "
            f"score={hit.score:.3f} lexical={hit.lexical_score:.3f} "
            f"semantic={hit.semantic_score:.3f} phrase={hit.phrase_score:.3f} "
            f"matched={', '.join(hit.matched_terms) or 'none'}\n{snippet}"
        )
    return "\n\n".join(parts)


@tool
def list_documents() -> str:
    """List uploaded documents available for document retrieval."""
    documents = get_document_store().documents()
    if not documents:
        return "No documents have been uploaded yet."
    return "\n".join(
        f"- {doc['title']} ({doc['doc_id']}): {doc['chunk_count']} chunks, uploaded {doc['uploaded']}"
        for doc in documents
    )


DOCUMENT_RAG_TOOLS = [upload_document, search_documents, list_documents]
