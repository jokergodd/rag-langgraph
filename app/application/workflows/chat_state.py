from typing import Annotated, Any, Literal, TypedDict

from langchain_core.documents import Document


def merge_list(left: list | None, right: list | None) -> list:
    """用于并行节点场景下的列表字段合并。"""
    return [*(left or []), *(right or [])]


def merge_dict(left: dict | None, right: dict | None) -> dict:
    """用于并行节点场景下的字典字段合并。"""
    merged = dict(left or {})
    for key, value in (right or {}).items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


class ChatGraphState(TypedDict, total=False):
    """在线问答图的共享状态。"""

    request_id: str
    session_id: str
    tenant_id: str
    user_id: str

    # 用户原始问题与改写后的检索问题
    question: str
    rewritten_queries: list[str]

    # 访问范围与检索策略
    access_scope: dict[str, Any]
    retrieval_policy: dict[str, Any]

    # 检索中间结果
    dense_ranked_lists: list[list[Document]]
    sparse_ranked_lists: list[list[Document]]
    fused_child_docs: list[Document]
    reranked_child_docs: list[Document]
    diversified_child_docs: list[Document]
    parent_docs: list[Document]

    # 生成阶段结果
    final_context: str
    answer: str
    citations: list[dict[str, Any]]

    # 并行节点会同时写入这些字段，因此需要 reducer 合并
    errors: Annotated[list[dict[str, Any]], merge_list]
    warnings: Annotated[list[str], merge_list]
    metrics: Annotated[dict[str, Any], merge_dict]
    debug: Annotated[dict[str, Any], merge_dict]

    # 路由控制字段
    route: Literal["answer", "fallback", "deny"]
