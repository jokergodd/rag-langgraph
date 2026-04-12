from typing import Any, Literal, TypedDict

from langchain_core.documents import Document


class ChatGraphState(TypedDict, total=False):
    request_id: str
    session_id: str
    tenant_id: str
    user_id: str

    question: str
    rewritten_queries: list[str]

    access_scope: dict[str, Any]
    retrieval_policy: dict[str, Any]

    dense_ranked_lists: list[list[Document]]
    sparse_ranked_lists: list[list[Document]]
    fused_child_docs: list[Document]
    reranked_child_docs: list[Document]
    diversified_child_docs: list[Document]
    parent_docs: list[Document]

    final_context: str
    answer: str
    citations: list[dict[str, Any]]

    errors: list[dict[str, Any]]
    warnings: list[str]
    metrics: dict[str, Any]
    debug: dict[str, Any]

    route: Literal["answer", "fallback", "deny"]

