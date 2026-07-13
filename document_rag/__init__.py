"""Document-only RAG for Caranthir.

This package is intentionally scoped to uploaded/readable documents. It is not
the long-term memory layer, but it is shaped so the retrieval backend can be
reused by memory later.
"""

from document_rag.store import DocumentRagStore, get_document_store, set_document_store
from document_rag.tools import DOCUMENT_RAG_TOOLS

__all__ = [
    "DOCUMENT_RAG_TOOLS",
    "DocumentRagStore",
    "get_document_store",
    "set_document_store",
]
