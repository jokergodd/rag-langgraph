from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

from app.application.bootstrap import initialize_rag_application
from app.application.workflows.chat_graph import summarize_state
from app.evaluation.deepeval_runner import run_deepeval_evaluation
from app.infrastructure.config.settings import settings
from app.infrastructure.observability.mlflow_tracker import MlflowTracker


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAG CLI with MLflow tracing and DeepEval evaluation.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("chat", help="Start the interactive RAG chat session.")

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run DeepEval on a JSON dataset and log results to MLflow.",
    )
    evaluate_parser.add_argument(
        "--dataset",
        type=Path,
        default=settings.deepeval_dataset_path,
        help="Path to the DeepEval dataset JSON file.",
    )

    return parser


def _print_build_summary(tracker: MlflowTracker, app) -> None:
    stats = app.build_stats
    print("Index build complete.")
    print(f"- raw docs: {stats.raw_doc_count}")
    print(f"- parent chunks: {stats.parent_chunk_count}")
    print(f"- child chunks: {stats.child_chunk_count}")
    print(f"- build time: {stats.total_build_ms} ms")
    if tracker.available:
        print(
            f"- mlflow: enabled ({settings.mlflow_tracking_uri}, "
            f"experiment={settings.mlflow_experiment_name})"
        )
    elif tracker.error:
        print(f"- mlflow: disabled ({tracker.error})")


def _run_chat_mode() -> None:
    if not settings.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {settings.data_dir.resolve()}")

    tracker = MlflowTracker(settings)
    with tracker.start_run(
        "rag-index-build",
        tags={"run_type": "index_build"},
    ):
        app = initialize_rag_application(settings, tracker)

    _print_build_summary(tracker, app)

    while True:
        question = input("\nEnter a question (type exit to quit): ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        request_payload = {"question": question}
        request_started_at = perf_counter()
        with tracker.start_run(
            "rag-chat-request",
            tags={"run_type": "chat_request"},
        ):
            tracker.log_params({"question": question})
            span_context = (
                tracker.start_span(
                    "chat_request",
                    span_type="CHAIN",
                    attributes={"question": question},
                )
                if settings.mlflow_trace_requests
                else nullcontext(None)
            )
            with span_context as span:
                result = app.chat_graph.invoke(request_payload)
                duration_ms = round((perf_counter() - request_started_at) * 1000, 2)

                if span:
                    span.set_inputs(request_payload)
                    span.set_outputs(
                        {
                            "answer": result.get("answer", ""),
                            "citations": result.get("citations", []),
                            "metrics": result.get("metrics", {}),
                        }
                    )
                tracker.log_metrics(
                    {
                        "request_duration_ms": duration_ms,
                        "parent_doc_count": len(result.get("parent_docs", [])),
                        "warning_count": len(result.get("warnings", [])),
                        "error_count": len(result.get("errors", [])),
                    }
                    | result.get("metrics", {})
                )
                tracker.log_dict(
                    {
                        "question": question,
                        "answer": result.get("answer", ""),
                        "citations": result.get("citations", []),
                        "warnings": result.get("warnings", []),
                        "errors": result.get("errors", []),
                        "metrics": result.get("metrics", {}),
                    },
                    "chat_request.json",
                )

        print("\n===== LangGraph Summary =====")
        print(summarize_state(result))

        print("\n===== Final Parent Context =====")
        docs = result.get("parent_docs", [])
        for index, doc in enumerate(docs, 1):
            print(
                f"{index}. source={doc.metadata.get('source_file')} "
                f"parent_id={doc.metadata.get('parent_id')}\n"
                f"{doc.page_content[:300]}\n"
            )

        print("\n===== Final Answer =====")
        print(result.get("answer", "Unable to answer from the retrieved context."))

        if result.get("warnings"):
            print("\n===== Warnings =====")
            for warning in result["warnings"]:
                print(f"- {warning}")


def _run_evaluate_mode(dataset: Path) -> None:
    if not dataset.exists():
        raise FileNotFoundError(f"DeepEval dataset not found: {dataset.resolve()}")

    tracker = MlflowTracker(settings)
    with tracker.start_run(
        "rag-deepeval",
        tags={"run_type": "deepeval_evaluation"},
    ):
        app = initialize_rag_application(settings, tracker)
        summary = run_deepeval_evaluation(
            app=app,
            settings=settings,
            tracker=tracker,
            dataset_path=dataset,
        )

    _print_build_summary(tracker, app)
    print("\n===== DeepEval Summary =====")
    print(f"dataset: {summary.dataset_path}")
    print(f"cases: {summary.case_count}")
    print(f"total runtime: {summary.total_runtime_ms} ms")
    for name, score in summary.average_scores.items():
        print(f"- {name}: {score}")


def run_cli() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    command = args.command or "chat"

    if command == "evaluate":
        _run_evaluate_mode(args.dataset)
        return

    _run_chat_mode()
