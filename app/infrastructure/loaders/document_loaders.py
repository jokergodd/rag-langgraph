from pathlib import Path

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document


def load_one_file(file_path: Path) -> list[Document]:
    suffix = file_path.suffix.lower()

    match suffix:
        case ".pdf":
            loader = PyPDFLoader(str(file_path))
        case ".md":
            loader = UnstructuredMarkdownLoader(str(file_path))
        case ".docx":
            loader = Docx2txtLoader(str(file_path))
        case _:
            return []

    docs = loader.load()
    for doc in docs:
        doc.metadata["source_file"] = file_path.name
        doc.metadata["source_path"] = str(file_path)
        doc.metadata["file_type"] = suffix.replace(".", "")
    return docs


def load_all_documents(data_dir: Path) -> list[Document]:
    all_docs: list[Document] = []
    for file_path in data_dir.rglob("*"):
        if file_path.is_file():
            all_docs.extend(load_one_file(file_path))
    return all_docs

