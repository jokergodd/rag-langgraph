import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("./data")
    chroma_dir: str = "./chroma_db"
    collection_name: str = "rag_demo_docs_parent_child_v2"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    llm_model_id: str | None = os.getenv("LLM_MODEL_ID")
    llm_api_key: str | None = os.getenv("LLM_API_KEY")
    llm_base_url: str | None = os.getenv("LLM_BASE_URL")
    llm_provider: str = "deepseek"
    llm_temperature: float = 0

    parent_chunk_size: int = 1200
    parent_chunk_overlap: int = 200
    child_chunk_size: int = 300
    child_chunk_overlap: int = 50

    dense_k: int = 8
    bm25_k: int = 8
    rrf_k: int = 60
    rerank_top_n: int = 12
    mmr_final_k: int = 6
    final_parent_k: int = 4

    mlflow_enabled: bool = _get_bool("MLFLOW_ENABLED", True)
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow_experiment_name: str = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        "study-python-rag",
    )
    mlflow_trace_requests: bool = _get_bool("MLFLOW_TRACE_REQUESTS", True)

    deepeval_model_id: str | None = os.getenv("DEEPEVAL_MODEL_ID") or os.getenv(
        "DEEPSEEK_MODEL_NAME"
    ) or os.getenv("LLM_MODEL_ID")
    deepeval_api_key: str | None = os.getenv("DEEPEVAL_API_KEY") or os.getenv(
        "DEEPSEEK_API_KEY"
    ) or os.getenv("LLM_API_KEY")
    deepeval_dataset_path: Path = Path(
        os.getenv("DEEPEVAL_DATASET_PATH", "./docs/rag_eval_dataset.example.json")
    )
    deepeval_threshold: float = float(os.getenv("DEEPEVAL_THRESHOLD", "0.5"))


settings = Settings()
