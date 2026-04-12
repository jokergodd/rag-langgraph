import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("./data")
    chroma_dir: str = "./chroma_db"
    collection_name: str = "rag_demo_docs_parent_child_v2"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    llm_model_id: str | None = os.getenv("LLM_MODEL_ID")
    llm_api_key: str | None = os.getenv("LLM_API_KEY")
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


settings = Settings()

