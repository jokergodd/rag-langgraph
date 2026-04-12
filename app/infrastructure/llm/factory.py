from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings

from app.infrastructure.config.settings import Settings


def build_chat_model(settings: Settings):
    return init_chat_model(
        model=settings.llm_model_id,
        api_key=settings.llm_api_key,
        model_provider=settings.llm_provider,
        temperature=settings.llm_temperature,
    )


def build_embedding_model(settings: Settings) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.embed_model)

