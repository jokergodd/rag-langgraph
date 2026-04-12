from dataclasses import dataclass
from time import perf_counter
from uuid import uuid4

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from app.application.workflows.chat_state import ChatGraphState
from app.domain.services.answer_service import generate_answer, render_context
from app.domain.services.ranking_service import (
    apply_mmr,
    expand_to_parents,
    format_docs,
    reciprocal_rank_fusion,
)
from app.domain.services.retrieval_service import rerank_child_docs, rewrite_queries
from app.infrastructure.config.settings import Settings


@dataclass(frozen=True)
class ChatGraphRuntime:
    model: object
    embedding_model: object
    dense_retriever: object
    bm25_retriever: object
    parent_store: dict[str, Document]
    settings: Settings


def _append_error(state: ChatGraphState, stage: str, exc: Exception) -> list[dict]:
    errors = list(state.get("errors", []))
    errors.append(
        {
            "stage": stage,
            "type": type(exc).__name__,
            "message": str(exc),
        }
    )
    return errors


def _base_metrics(state: ChatGraphState) -> dict:
    metrics = dict(state.get("metrics", {}))
    node_timings = dict(metrics.get("node_timings_ms", {}))
    metrics["node_timings_ms"] = node_timings
    return metrics


def _time_node(state: ChatGraphState, node_name: str, started_at: float) -> dict:
    metrics = _base_metrics(state)
    metrics["node_timings_ms"][node_name] = round((perf_counter() - started_at) * 1000, 2)
    return metrics


def build_chat_graph(runtime: ChatGraphRuntime):
    def init_request(state: ChatGraphState) -> ChatGraphState:
        started_at = perf_counter()
        next_state: ChatGraphState = {
            "request_id": state.get("request_id", str(uuid4())),
            "session_id": state.get("session_id", "cli-session"),
            "tenant_id": state.get("tenant_id", "default-tenant"),
            "user_id": state.get("user_id", "default-user"),
            "access_scope": state.get("access_scope", {"mode": "single-tenant"}),
            "retrieval_policy": state.get(
                "retrieval_policy",
                {
                    "dense_k": runtime.settings.dense_k,
                    "bm25_k": runtime.settings.bm25_k,
                    "rrf_k": runtime.settings.rrf_k,
                    "rerank_top_n": runtime.settings.rerank_top_n,
                    "mmr_final_k": runtime.settings.mmr_final_k,
                    "final_parent_k": runtime.settings.final_parent_k,
                },
            ),
            "errors": list(state.get("errors", [])),
            "warnings": list(state.get("warnings", [])),
            "debug": dict(state.get("debug", {})),
        }
        next_state["metrics"] = _time_node(next_state, "init_request", started_at)
        return next_state

    def authorize_request(state: ChatGraphState) -> ChatGraphState:
        started_at = perf_counter()
        next_state: ChatGraphState = {
            "route": "answer",
            "access_scope": state["access_scope"],
            "warnings": list(state.get("warnings", [])),
        }
        next_state["metrics"] = _time_node(state | next_state, "authorize_request", started_at)
        return next_state

    def rewrite_query_node(state: ChatGraphState) -> ChatGraphState:
        started_at = perf_counter()
        try:
            rewritten_queries = rewrite_queries(state["question"], runtime.model)
            next_state: ChatGraphState = {"rewritten_queries": rewritten_queries}
        except Exception as exc:
            next_state = {
                "rewritten_queries": [state["question"]],
                "errors": _append_error(state, "rewrite_query", exc),
                "warnings": list(state.get("warnings", []))
                + ["query rewrite failed, fallback to original question"],
            }
        next_state["metrics"] = _time_node(state | next_state, "rewrite_query", started_at)
        return next_state

    def retrieve_dense_node(state: ChatGraphState) -> ChatGraphState:
        started_at = perf_counter()
        try:
            dense_ranked_lists = [
                runtime.dense_retriever.invoke(query)
                for query in state.get("rewritten_queries", [state["question"]])
            ]
            next_state: ChatGraphState = {"dense_ranked_lists": dense_ranked_lists}
        except Exception as exc:
            next_state = {
                "dense_ranked_lists": [],
                "errors": _append_error(state, "retrieve_dense", exc),
                "warnings": list(state.get("warnings", []))
                + ["dense retrieval failed, continue with sparse retrieval"],
            }
        next_state["metrics"] = _time_node(state | next_state, "retrieve_dense", started_at)
        return next_state

    def retrieve_sparse_node(state: ChatGraphState) -> ChatGraphState:
        started_at = perf_counter()
        try:
            sparse_ranked_lists = [
                runtime.bm25_retriever.invoke(query)
                for query in state.get("rewritten_queries", [state["question"]])
            ]
            next_state: ChatGraphState = {"sparse_ranked_lists": sparse_ranked_lists}
        except Exception as exc:
            next_state = {
                "sparse_ranked_lists": [],
                "errors": _append_error(state, "retrieve_sparse", exc),
                "warnings": list(state.get("warnings", []))
                + ["sparse retrieval failed, continue with available results"],
            }
        next_state["metrics"] = _time_node(state | next_state, "retrieve_sparse", started_at)
        return next_state

    def fuse_results_node(state: ChatGraphState) -> ChatGraphState:
        started_at = perf_counter()
        ranked_lists = state.get("dense_ranked_lists", []) + state.get("sparse_ranked_lists", [])
        fused_child_docs = reciprocal_rank_fusion(ranked_lists, rrf_k=runtime.settings.rrf_k)
        next_state: ChatGraphState = {
            "fused_child_docs": fused_child_docs,
            "debug": {
                **state.get("debug", {}),
                "dense_query_count": len(state.get("dense_ranked_lists", [])),
                "sparse_query_count": len(state.get("sparse_ranked_lists", [])),
                "fused_child_count": len(fused_child_docs),
            },
        }
        next_state["metrics"] = _time_node(state | next_state, "fuse_results", started_at)
        return next_state

    def rerank_results_node(state: ChatGraphState) -> ChatGraphState:
        started_at = perf_counter()
        try:
            reranked_child_docs = rerank_child_docs(
                state["question"],
                state.get("fused_child_docs", []),
                runtime.settings,
            )
            next_state: ChatGraphState = {"reranked_child_docs": reranked_child_docs}
        except Exception as exc:
            next_state = {
                "reranked_child_docs": list(state.get("fused_child_docs", [])),
                "errors": _append_error(state, "rerank_results", exc),
                "warnings": list(state.get("warnings", []))
                + ["rerank failed, fallback to fused results"],
            }
        next_state["metrics"] = _time_node(state | next_state, "rerank_results", started_at)
        return next_state

    def diversify_results_node(state: ChatGraphState) -> ChatGraphState:
        started_at = perf_counter()
        try:
            diversified_child_docs = apply_mmr(
                question=state["question"],
                docs=state.get("reranked_child_docs", []),
                embedding_model=runtime.embedding_model,
                lambda_mult=0.7,
                final_k=runtime.settings.mmr_final_k,
            )
            next_state: ChatGraphState = {"diversified_child_docs": diversified_child_docs}
        except Exception as exc:
            next_state = {
                "diversified_child_docs": list(state.get("reranked_child_docs", [])),
                "errors": _append_error(state, "diversify_results", exc),
                "warnings": list(state.get("warnings", []))
                + ["mmr failed, fallback to reranked results"],
            }
        next_state["metrics"] = _time_node(state | next_state, "diversify_results", started_at)
        return next_state

    def expand_parent_context_node(state: ChatGraphState) -> ChatGraphState:
        started_at = perf_counter()
        parent_docs = expand_to_parents(
            child_docs=state.get("diversified_child_docs", []),
            parent_store=runtime.parent_store,
            final_parent_k=runtime.settings.final_parent_k,
        )
        next_state: ChatGraphState = {
            "parent_docs": parent_docs,
            "debug": {
                **state.get("debug", {}),
                "parent_doc_count": len(parent_docs),
            },
        }
        next_state["metrics"] = _time_node(
            state | next_state, "expand_parent_context", started_at
        )
        return next_state

    def build_prompt_context_node(state: ChatGraphState) -> ChatGraphState:
        started_at = perf_counter()
        parent_docs = state.get("parent_docs", [])
        citations = [
            {
                "source_file": doc.metadata.get("source_file", "unknown"),
                "parent_id": doc.metadata.get("parent_id", "NA"),
            }
            for doc in parent_docs
        ]
        next_state: ChatGraphState = {
            "final_context": render_context(parent_docs),
            "citations": citations,
        }
        next_state["metrics"] = _time_node(
            state | next_state, "build_prompt_context", started_at
        )
        return next_state

    def decide_answer_route_node(state: ChatGraphState) -> ChatGraphState:
        started_at = perf_counter()
        if state.get("route") == "deny":
            route = "deny"
        elif state.get("parent_docs"):
            route = "answer"
        else:
            route = "fallback"
        next_state: ChatGraphState = {"route": route}
        next_state["metrics"] = _time_node(
            state | next_state, "decide_answer_route", started_at
        )
        return next_state

    def generate_answer_node(state: ChatGraphState) -> ChatGraphState:
        started_at = perf_counter()
        try:
            answer = generate_answer(
                question=state["question"],
                context=state.get("final_context", ""),
                model=runtime.model,
            )
            next_state: ChatGraphState = {"answer": answer}
        except Exception as exc:
            next_state = {
                "answer": "根据当前检索到的资料，我不能确定。",
                "errors": _append_error(state, "generate_answer", exc),
                "warnings": list(state.get("warnings", []))
                + ["answer generation failed, fallback answer returned"],
            }
        next_state["metrics"] = _time_node(state | next_state, "generate_answer", started_at)
        return next_state

    def fallback_answer_node(state: ChatGraphState) -> ChatGraphState:
        started_at = perf_counter()
        if state.get("route") == "deny":
            answer = "当前请求没有足够权限访问相关知识内容。"
        elif state.get("errors"):
            answer = "根据当前检索到的资料，我不能确定。部分检索或生成步骤已降级处理。"
        else:
            answer = "根据当前检索到的资料，我不能确定。"
        next_state: ChatGraphState = {"answer": answer}
        next_state["metrics"] = _time_node(state | next_state, "fallback_answer", started_at)
        return next_state

    def record_observation_node(state: ChatGraphState) -> ChatGraphState:
        started_at = perf_counter()
        debug = dict(state.get("debug", {}))
        debug["rewritten_queries"] = state.get("rewritten_queries", [])
        debug["fused_child_count"] = len(state.get("fused_child_docs", []))
        debug["reranked_child_count"] = len(state.get("reranked_child_docs", []))
        debug["diversified_child_count"] = len(state.get("diversified_child_docs", []))
        debug["parent_doc_count"] = len(state.get("parent_docs", []))

        metrics = _base_metrics(state)
        metrics["request_question_length"] = len(state.get("question", ""))
        metrics["final_context_length"] = len(state.get("final_context", ""))
        metrics["error_count"] = len(state.get("errors", []))
        metrics["warning_count"] = len(state.get("warnings", []))
        metrics["node_timings_ms"]["record_observation"] = round(
            (perf_counter() - started_at) * 1000, 2
        )

        return {
            "debug": debug,
            "metrics": metrics,
        }

    def route_after_decision(state: ChatGraphState) -> str:
        return state.get("route", "fallback")

    graph = StateGraph(ChatGraphState)
    graph.add_node("init_request", init_request)
    graph.add_node("authorize_request", authorize_request)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("retrieve_dense", retrieve_dense_node)
    graph.add_node("retrieve_sparse", retrieve_sparse_node)
    graph.add_node("fuse_results", fuse_results_node)
    graph.add_node("rerank_results", rerank_results_node)
    graph.add_node("diversify_results", diversify_results_node)
    graph.add_node("expand_parent_context", expand_parent_context_node)
    graph.add_node("build_prompt_context", build_prompt_context_node)
    graph.add_node("decide_answer_route", decide_answer_route_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("fallback_answer", fallback_answer_node)
    graph.add_node("record_observation", record_observation_node)

    graph.add_edge(START, "init_request")
    graph.add_edge("init_request", "authorize_request")
    graph.add_edge("authorize_request", "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve_dense")
    graph.add_edge("retrieve_dense", "retrieve_sparse")
    graph.add_edge("retrieve_sparse", "fuse_results")
    graph.add_edge("fuse_results", "rerank_results")
    graph.add_edge("rerank_results", "diversify_results")
    graph.add_edge("diversify_results", "expand_parent_context")
    graph.add_edge("expand_parent_context", "build_prompt_context")
    graph.add_edge("build_prompt_context", "decide_answer_route")
    graph.add_conditional_edges(
        "decide_answer_route",
        route_after_decision,
        {
            "answer": "generate_answer",
            "fallback": "fallback_answer",
            "deny": "fallback_answer",
        },
    )
    graph.add_edge("generate_answer", "record_observation")
    graph.add_edge("fallback_answer", "record_observation")
    graph.add_edge("record_observation", END)

    return graph.compile()


def summarize_state(state: ChatGraphState) -> str:
    lines = []
    for index, query in enumerate(state.get("rewritten_queries", []), 1):
        lines.append(f"{index}. {query}")

    lines.append("")
    lines.append(f"[RRF 后 child 数] {len(state.get('fused_child_docs', []))}")
    lines.append(f"[Rerank 后 child 数] {len(state.get('reranked_child_docs', []))}")
    lines.append(f"[MMR 后 child 数] {len(state.get('diversified_child_docs', []))}")
    lines.append(f"[最终 parent 数] {len(state.get('parent_docs', []))}")
    return "\n".join(lines)


def format_parent_docs_preview(docs: list[Document]) -> str:
    if not docs:
        return ""
    return format_docs(docs)

