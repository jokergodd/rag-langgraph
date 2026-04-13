from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

from app.application.workflows.chat_graph import ChatGraphRuntime, build_chat_graph
from app.domain.services.chunking_service import (
    build_parent_store,
    split_child_documents,
    split_parent_documents,
)
from app.domain.services.retrieval_service import build_child_retrievers
from app.infrastructure.config.settings import Settings
from app.infrastructure.llm.factory import build_chat_model, build_embedding_model
from app.infrastructure.loaders.document_loaders import load_all_documents
from app.infrastructure.observability.mlflow_tracker import MlflowTracker
from app.infrastructure.vectorstores.chroma_store import build_vectorstore


@dataclass(frozen=True)
class IndexBuildStats:
    raw_doc_count: int
    parent_chunk_count: int
    child_chunk_count: int
    load_documents_ms: float
    split_parent_ms: float
    split_child_ms: float
    build_vectorstore_ms: float
    build_retrievers_ms: float
    total_build_ms: float

    def metrics(self) -> dict[str, float]:
        return {
            "raw_doc_count": self.raw_doc_count,
            "parent_chunk_count": self.parent_chunk_count,
            "child_chunk_count": self.child_chunk_count,
            "load_documents_ms": self.load_documents_ms,
            "split_parent_ms": self.split_parent_ms,
            "split_child_ms": self.split_child_ms,
            "build_vectorstore_ms": self.build_vectorstore_ms,
            "build_retrievers_ms": self.build_retrievers_ms,
            "total_build_ms": self.total_build_ms,
        }


@dataclass(frozen=True)
class RagApplication:
    chat_graph: object
    runtime: ChatGraphRuntime
    build_stats: IndexBuildStats


def initialize_rag_application(
    settings: Settings,
    tracker: MlflowTracker | None = None,
) -> RagApplication:
    total_started_at = perf_counter()
    tracker = tracker or MlflowTracker(settings)

    chat_model = build_chat_model(settings)
    embedding_model = build_embedding_model(settings)

    with tracker.start_span("load_documents", span_type="RETRIEVER") as span:
        started_at = perf_counter()
        raw_docs = load_all_documents(settings.data_dir)
        load_documents_ms = round((perf_counter() - started_at) * 1000, 2)
        if span:
            span.set_outputs({"raw_doc_count": len(raw_docs)})

    with tracker.start_span("split_parent_documents", span_type="CHAIN") as span:
        started_at = perf_counter()
        parent_docs = split_parent_documents(raw_docs, settings)
        split_parent_ms = round((perf_counter() - started_at) * 1000, 2)
        if span:
            span.set_outputs({"parent_chunk_count": len(parent_docs)})

    with tracker.start_span("split_child_documents", span_type="CHAIN") as span:
        started_at = perf_counter()
        child_docs = split_child_documents(parent_docs, settings)
        split_child_ms = round((perf_counter() - started_at) * 1000, 2)
        if span:
            span.set_outputs({"child_chunk_count": len(child_docs)})

    parent_store = build_parent_store(parent_docs)

    with tracker.start_span("build_vectorstore", span_type="RETRIEVER") as span:
        started_at = perf_counter()
        vectorstore = build_vectorstore(child_docs, embedding_model, settings)
        build_vectorstore_ms = round((perf_counter() - started_at) * 1000, 2)
        if span:
            span.set_outputs({"collection_name": settings.collection_name})

    with tracker.start_span("build_retrievers", span_type="RETRIEVER") as span:
        started_at = perf_counter()
        dense_retriever, bm25_retriever = build_child_retrievers(
            child_docs,
            vectorstore,
            settings,
        )
        build_retrievers_ms = round((perf_counter() - started_at) * 1000, 2)
        if span:
            span.set_outputs(
                {
                    "dense_k": settings.dense_k,
                    "bm25_k": settings.bm25_k,
                }
            )

    runtime = ChatGraphRuntime(
        model=chat_model,
        embedding_model=embedding_model,
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        parent_store=parent_store,
        settings=settings,
    )

    build_stats = IndexBuildStats(
        raw_doc_count=len(raw_docs),
        parent_chunk_count=len(parent_docs),
        child_chunk_count=len(child_docs),
        load_documents_ms=load_documents_ms,
        split_parent_ms=split_parent_ms,
        split_child_ms=split_child_ms,
        build_vectorstore_ms=build_vectorstore_ms,
        build_retrievers_ms=build_retrievers_ms,
        total_build_ms=round((perf_counter() - total_started_at) * 1000, 2),
    )

    tracker.log_metrics(build_stats.metrics())
    tracker.log_params(
        {
            "collection_name": settings.collection_name,
            "embed_model": settings.embed_model,
            "parent_chunk_size": settings.parent_chunk_size,
            "parent_chunk_overlap": settings.parent_chunk_overlap,
            "child_chunk_size": settings.child_chunk_size,
            "child_chunk_overlap": settings.child_chunk_overlap,
            "dense_k": settings.dense_k,
            "bm25_k": settings.bm25_k,
            "rerank_top_n": settings.rerank_top_n,
            "mmr_final_k": settings.mmr_final_k,
            "final_parent_k": settings.final_parent_k,
        }
    )
    tracker.log_dict(asdict(build_stats), "index_build_stats.json")

    return RagApplication(
        chat_graph=build_chat_graph(runtime),
        runtime=runtime,
        build_stats=build_stats,
    )
