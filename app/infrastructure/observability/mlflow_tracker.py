from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from app.infrastructure.config.settings import Settings


def _normalize_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _normalize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_value(item) for item in value]
    return str(value)


class MlflowTracker:
    """Thin wrapper around MLflow with a safe no-op fallback."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = settings.mlflow_enabled
        self._mlflow = None
        self.error: str | None = None

        if not self.enabled:
            return

        try:
            import mlflow

            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment(settings.mlflow_experiment_name)
            self._mlflow = mlflow
        except Exception as exc:
            self.enabled = False
            self.error = str(exc)

    @property
    def available(self) -> bool:
        return self.enabled and self._mlflow is not None

    @contextmanager
    def start_run(
        self,
        run_name: str,
        tags: dict[str, Any] | None = None,
    ):
        if not self.available:
            yield None
            return

        with self._mlflow.start_run(run_name=run_name):
            if tags:
                self._mlflow.set_tags(_normalize_value(tags))
            yield self._mlflow.active_run()

    def active_run(self):
        if not self.available:
            return None
        return self._mlflow.active_run()

    @contextmanager
    def start_span(
        self,
        name: str,
        span_type: str = "UNKNOWN",
        attributes: dict[str, Any] | None = None,
    ):
        if not self.available:
            yield None
            return

        with self._mlflow.start_span(
            name=name,
            span_type=span_type,
            attributes=_normalize_value(attributes or {}),
        ) as span:
            yield span

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        if not self.available or not metrics:
            return
        normalized = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and value is not None
        }
        if normalized:
            self._mlflow.log_metrics(normalized)

    def log_params(self, params: dict[str, Any]) -> None:
        if not self.available or not params:
            return
        normalized = {
            key: str(value) for key, value in params.items() if value is not None
        }
        if normalized:
            self._mlflow.log_params(normalized)

    def log_dict(self, payload: dict[str, Any] | list[Any], artifact_file: str) -> None:
        if not self.available:
            return
        self._mlflow.log_dict(_normalize_value(payload), artifact_file)

    def set_tags(self, tags: dict[str, Any]) -> None:
        if not self.available or not tags:
            return
        self._mlflow.set_tags(_normalize_value(tags))
