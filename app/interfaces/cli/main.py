from app.application.workflows.chat_graph import (
    ChatGraphRuntime,
    build_chat_graph,
    summarize_state,
)
from app.domain.services.chunking_service import (
    build_parent_store,
    split_child_documents,
    split_parent_documents,
)
from app.domain.services.retrieval_service import build_child_retrievers
from app.infrastructure.config.settings import settings
from app.infrastructure.llm.factory import build_chat_model, build_embedding_model
from app.infrastructure.loaders.document_loaders import load_all_documents
from app.infrastructure.vectorstores.chroma_store import build_vectorstore


def run_cli() -> None:
    """命令行入口：初始化索引后进入交互式问答循环。"""
    if not settings.data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在：{settings.data_dir.resolve()}")

    chat_model = build_chat_model(settings)
    embedding_model = build_embedding_model(settings)

    print("1) 加载文档...")
    raw_docs = load_all_documents(settings.data_dir)
    print(f"原始文档数：{len(raw_docs)}")

    print("2) 切分父片段...")
    parent_docs = split_parent_documents(raw_docs, settings)
    print(f"父片段数：{len(parent_docs)}")

    print("3) 切分子片段...")
    child_docs = split_child_documents(parent_docs, settings)
    print(f"子片段数：{len(child_docs)}")

    parent_store = build_parent_store(parent_docs)

    print("4) 构建向量库...")
    vectorstore = build_vectorstore(child_docs, embedding_model, settings)

    print("5) 构建子片段检索器...")
    dense_retriever, bm25_retriever = build_child_retrievers(
        child_docs,
        vectorstore,
        settings,
    )

    print("6) 构建 LangGraph 中文问答工作流...")
    runtime = ChatGraphRuntime(
        model=chat_model,
        embedding_model=embedding_model,
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        parent_store=parent_store,
        settings=settings,
    )
    chat_graph = build_chat_graph(runtime)

    while True:
        question = input("\n请输入问题（输入 exit 退出）：").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        result = chat_graph.invoke({"question": question})

        print("\n===== LangGraph 执行摘要 =====")
        print(summarize_state(result))

        print("\n===== 最终父片段上下文 =====")
        docs = result.get("parent_docs", [])
        for index, doc in enumerate(docs, 1):
            print(
                f"{index}. 来源文件={doc.metadata.get('source_file')} "
                f"父片段ID={doc.metadata.get('parent_id')}\n"
                f"{doc.page_content[:300]}\n"
            )

        print("\n===== 最终答案 =====")
        print(result.get("answer", "根据当前检索到的资料，我不能确定。"))

        if result.get("warnings"):
            print("\n===== 警告 =====")
            for warning in result["warnings"]:
                print(f"- {warning}")
