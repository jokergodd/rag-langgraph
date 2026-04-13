from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models import DeepSeekModel
from deepeval.test_case import LLMTestCase

from app.application.bootstrap import RagApplication
from app.infrastructure.config.settings import Settings
from app.infrastructure.observability.mlflow_tracker import MlflowTracker


@dataclass(frozen=True)
class EvaluationSummary:
    dataset_path: str
    case_count: int
    average_scores: dict[str, float]
    case_results: list[dict[str, Any]]
    total_runtime_ms: float


def load_eval_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("DeepEval dataset must be a JSON array.")
    return payload


def _build_deepeval_model(settings: Settings) -> DeepSeekModel:
    if not settings.deepeval_model_id:
        raise ValueError("DEEPEVAL_MODEL_ID or LLM_MODEL_ID is required for DeepEval.")
    if not settings.deepeval_api_key:
        raise ValueError("DEEPEVAL_API_KEY, DEEPSEEK_API_KEY, or LLM_API_KEY is required.")
    return DeepSeekModel(
        model=settings.deepeval_model_id,
        api_key=settings.deepeval_api_key,
        temperature=settings.llm_temperature,
    )


def _build_metrics(settings: Settings) -> list[Any]:
    model = _build_deepeval_model(settings)
    common = {
        "threshold": settings.deepeval_threshold,
        "model": model,
        "include_reason": True,
        "async_mode": False,
    }
    return [
        AnswerRelevancyMetric(**common),
        FaithfulnessMetric(**common),
        ContextualPrecisionMetric(**common),
        ContextualRecallMetric(**common),
        ContextualRelevancyMetric(**common),
    ]


def run_deepeval_evaluation(
    app: RagApplication,
    settings: Settings,
    tracker: MlflowTracker,
    dataset_path: Path | None = None,
) -> EvaluationSummary:
    dataset_path = dataset_path or settings.deepeval_dataset_path
    cases = load_eval_dataset(dataset_path)
    metrics = _build_metrics(settings)

    started_at = perf_counter()
    test_cases: list[LLMTestCase] = []
    raw_case_results: list[dict[str, Any]] = []

    for case in cases:
        question = case["question"]
        request_payload = {
            "question": question,
            "session_id": "deepeval-session",
            "user_id": "deepeval",
            "tenant_id": "offline-evaluation",
        }

        with tracker.start_span(
            "deepeval_case",
            span_type="EVALUATOR",
            attributes={
                "case_name": case.get("name", question[:60]),
                "question": question,
            },
        ) as span:
            case_started_at = perf_counter()
            result = app.chat_graph.invoke(request_payload)
            case_runtime_ms = round((perf_counter() - case_started_at) * 1000, 2)

            retrieved_context = [
                doc.page_content for doc in result.get("parent_docs", [])
            ]
            expected_context = case.get("expected_context") or []

            test_case = LLMTestCase(
                name=case.get("name"),
                input=question,
                actual_output=result.get("answer", ""),
                expected_output=case.get("expected_answer"),
                context=expected_context,
                retrieval_context=retrieved_context,
                additional_metadata={
                    "tags": case.get("tags", []),
                    "request_metrics": result.get("metrics", {}),
                    "citations": result.get("citations", []),
                },
            )
            test_cases.append(test_case)

            raw_case_result = {
                "name": case.get("name", question[:60]),
                "question": question,
                "expected_answer": case.get("expected_answer"),
                "expected_context": expected_context,
                "actual_answer": result.get("answer", ""),
                "retrieval_context": retrieved_context,
                "citations": result.get("citations", []),
                "request_metrics": result.get("metrics", {}),
                "runtime_ms": case_runtime_ms,
            }
            raw_case_results.append(raw_case_result)

            if span:
                span.set_outputs(
                    {
                        "runtime_ms": case_runtime_ms,
                        "retrieved_context_count": len(retrieved_context),
                        "warning_count": len(result.get("warnings", [])),
                    }
                )

    evaluation_result = evaluate(test_cases=test_cases, metrics=metrics)

    per_metric_scores: dict[str, list[float]] = defaultdict(list)
    case_results: list[dict[str, Any]] = []
    for raw_case_result, test_result in zip(raw_case_results, evaluation_result.test_results):
        metric_rows = []
        for metric_data in test_result.metrics_data or []:
            if metric_data.score is not None:
                per_metric_scores[metric_data.name].append(metric_data.score)
            metric_rows.append(
                {
                    "name": metric_data.name,
                    "score": metric_data.score,
                    "success": metric_data.success,
                    "reason": metric_data.reason,
                    "threshold": metric_data.threshold,
                    "evaluation_model": metric_data.evaluation_model,
                }
            )
        case_results.append(raw_case_result | {"metrics": metric_rows})

    average_scores = {
        metric_name: round(sum(scores) / len(scores), 4)
        for metric_name, scores in per_metric_scores.items()
        if scores
    }
    total_runtime_ms = round((perf_counter() - started_at) * 1000, 2)

    tracker.log_params(
        {
            "evaluation_dataset": str(dataset_path),
            "deepeval_model_id": settings.deepeval_model_id,
            "deepeval_threshold": settings.deepeval_threshold,
        }
    )
    tracker.log_metrics(
        average_scores
        | {
            "evaluation_case_count": len(case_results),
            "evaluation_total_runtime_ms": total_runtime_ms,
        }
    )
    tracker.log_dict(case_results, "deepeval_case_results.json")
    tracker.log_dict(
        {
            "average_scores": average_scores,
            "test_run_id": evaluation_result.test_run_id,
            "confident_link": evaluation_result.confident_link,
            "total_runtime_ms": total_runtime_ms,
        },
        "deepeval_summary.json",
    )

    return EvaluationSummary(
        dataset_path=str(dataset_path),
        case_count=len(case_results),
        average_scores=average_scores,
        case_results=case_results,
        total_runtime_ms=total_runtime_ms,
    )
