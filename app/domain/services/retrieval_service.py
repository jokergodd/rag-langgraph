from langchain_community.document_compressors import FlashrankRerank
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.domain.services.ranking_service import (
    apply_mmr,
    expand_to_parents,
    reciprocal_rank_fusion,
)
from app.infrastructure.config.settings import Settings


def build_child_retrievers(
    child_docs: list[Document],
    vectorstore,
    settings: Settings,
):
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": settings.dense_k})
    bm25_retriever = BM25Retriever.from_documents(child_docs, k=settings.bm25_k)
    return dense_retriever, bm25_retriever


def rerank_child_docs(
    question: str,
    docs: list[Document],
    settings: Settings,
) -> list[Document]:
    if not docs:
        return []

    reranker = FlashrankRerank(top_n=min(settings.rerank_top_n, len(docs)))
    compressed = reranker.compress_documents(docs, query=question)
    return list(compressed)


def rewrite_queries(question: str, model) -> list[str]:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_template(
        """你是检索改写助手。
请把用户问题改写成 3 个不同但等价的检索问法。
要求：
1. 保留原始含义，不要扩写新事实
2. 尽量覆盖不同表达方式
3. 每行只输出一个问法
4. 不要编号，不要解释

用户问题：
{question}
"""
    )
    chain = prompt | model | StrOutputParser()
    raw = chain.invoke({"question": question})
    rewrites = [line.strip() for line in raw.splitlines() if line.strip()]

    results = [question]
    for query in rewrites:
        if query not in results:
            results.append(query)
    return results[:4]


def retrieve_parent_context(
    question: str,
    dense_retriever,
    bm25_retriever,
    parent_store: dict[str, Document],
    embedding_model: HuggingFaceEmbeddings,
    model,
    settings: Settings,
) -> list[Document]:
    queries = rewrite_queries(question, model)
    print("\n[MultiQuery]")
    for index, query in enumerate(queries, 1):
        print(f"{index}. {query}")

    ranked_lists: list[list[Document]] = []
    for query in queries:
        dense_docs = dense_retriever.invoke(query)
        bm25_docs = bm25_retriever.invoke(query)
        ranked_lists.append(dense_docs)
        ranked_lists.append(bm25_docs)

    rrf_child_docs = reciprocal_rank_fusion(ranked_lists, rrf_k=settings.rrf_k)
    print(f"\n[RRF 后 child 数] {len(rrf_child_docs)}")

    reranked_child_docs = rerank_child_docs(question, rrf_child_docs, settings)
    print(f"[Rerank 后 child 数] {len(reranked_child_docs)}")

    mmr_child_docs = apply_mmr(
        question=question,
        docs=reranked_child_docs,
        embedding_model=embedding_model,
        lambda_mult=0.7,
        final_k=settings.mmr_final_k,
    )
    print(f"[MMR 后 child 数] {len(mmr_child_docs)}")

    parent_docs = expand_to_parents(
        child_docs=mmr_child_docs,
        parent_store=parent_store,
        final_parent_k=settings.final_parent_k,
    )
    print(f"[最终 parent 数] {len(parent_docs)}")
    return parent_docs

