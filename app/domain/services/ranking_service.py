from collections import defaultdict

import numpy as np
from langchain_core.documents import Document
from langchain_core.vectorstores.utils import maximal_marginal_relevance
from langchain_huggingface import HuggingFaceEmbeddings

from app.infrastructure.config.settings import Settings


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    rrf_k: int,
) -> list[Document]:
    """使用 RRF 对多路排序结果进行融合。"""
    scored: dict[str, tuple[float, Document]] = {}

    for docs in ranked_lists:
        for rank, doc in enumerate(docs, start=1):
            child_id = doc.metadata.get("child_id")
            if not child_id:
                continue

            score = 1.0 / (rrf_k + rank)
            if child_id not in scored:
                scored[child_id] = (score, doc)
            else:
                previous_score, previous_doc = scored[child_id]
                scored[child_id] = (previous_score + score, previous_doc)

    sorted_docs = sorted(
        scored.values(),
        key=lambda item: item[0],
        reverse=True,
    )
    return [doc for _, doc in sorted_docs]


def apply_mmr(
    question: str,
    docs: list[Document],
    embedding_model: HuggingFaceEmbeddings,
    lambda_mult: float = 0.7,
    final_k: int = Settings.mmr_final_k,
) -> list[Document]:
    """使用 MMR 在保证相关性的同时减少结果冗余。"""
    if len(docs) <= final_k:
        return docs

    query_vec = np.array(embedding_model.embed_query(question))
    doc_vecs = np.array(embedding_model.embed_documents([doc.page_content for doc in docs]))

    # 直接调用 LangChain 内置 MMR，返回被选中的下标列表
    selected_indices = maximal_marginal_relevance(
        query_embedding=query_vec,
        embedding_list=doc_vecs,
        lambda_mult=lambda_mult,
        k=final_k,
    )

    return [docs[index] for index in selected_indices]


def expand_to_parents(
    child_docs: list[Document],
    parent_store: dict[str, Document],
    final_parent_k: int,
) -> list[Document]:
    """将子片段命中聚合回父片段。"""
    parent_hits: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "first_rank": 10**9, "doc": None}
    )

    for rank, child in enumerate(child_docs, start=1):
        parent_id = child.metadata.get("parent_id")
        if not parent_id or parent_id not in parent_store:
            continue

        parent_hits[parent_id]["count"] += 1
        parent_hits[parent_id]["first_rank"] = min(
            parent_hits[parent_id]["first_rank"],
            rank,
        )
        parent_hits[parent_id]["doc"] = parent_store[parent_id]

    sorted_parents = sorted(
        parent_hits.values(),
        key=lambda item: (-item["count"], item["first_rank"]),
    )
    return [item["doc"] for item in sorted_parents[:final_parent_k]]


def format_docs(docs: list[Document]) -> str:
    """将文档片段格式化为中文上下文。"""
    lines = []
    for index, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "未知文件")
        parent_id = doc.metadata.get("parent_id", "未标记")
        lines.append(
            f"[片段{index}] 来源文件={source}，父片段ID={parent_id}\n{doc.page_content}"
        )
    return "\n\n".join(lines)
