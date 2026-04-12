from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.infrastructure.config.settings import Settings


def split_parent_documents(
    docs: list[Document],
    settings: Settings,
) -> list[Document]:
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.parent_chunk_size,
        chunk_overlap=settings.parent_chunk_overlap,
    )
    parent_docs = parent_splitter.split_documents(docs)

    for index, doc in enumerate(parent_docs):
        doc.metadata["parent_id"] = f"parent_{index}"

    return parent_docs


def split_child_documents(
    parent_docs: list[Document],
    settings: Settings,
) -> list[Document]:
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.child_chunk_size,
        chunk_overlap=settings.child_chunk_overlap,
    )

    child_docs: list[Document] = []

    for parent in parent_docs:
        parent_id = parent.metadata["parent_id"]
        source_file = parent.metadata.get("source_file", "")
        file_type = parent.metadata.get("file_type", "")

        pieces = child_splitter.split_documents([parent])
        for index, child in enumerate(pieces):
            child.metadata["parent_id"] = parent_id
            child.metadata["child_id"] = f"{parent_id}_child_{index}"
            child.metadata["source_file"] = source_file
            child.metadata["file_type"] = file_type
            child.page_content = (
                f"来源文件: {source_file}\n文件类型: {file_type}\n{child.page_content}"
            )
            child_docs.append(child)

    return child_docs


def build_parent_store(parent_docs: list[Document]) -> dict[str, Document]:
    return {doc.metadata["parent_id"]: doc for doc in parent_docs}

