from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.infrastructure.config.settings import Settings


def build_vectorstore(
    child_docs: list[Document],
    embedding_model: HuggingFaceEmbeddings,
    settings: Settings,
) -> Chroma:
    vectorstore = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embedding_model,
        persist_directory=settings.chroma_dir,
    )

    try:
        vectorstore.delete_collection()
    except Exception:
        pass

    vectorstore = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embedding_model,
        persist_directory=settings.chroma_dir,
    )

    if child_docs:
        vectorstore.add_documents(child_docs)

    return vectorstore

