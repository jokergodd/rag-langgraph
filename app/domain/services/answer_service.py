from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from app.domain.services.ranking_service import format_docs


def build_answer_prompt() -> ChatPromptTemplate:
    """构建问答阶段的中文提示词。"""
    return ChatPromptTemplate.from_template(
        """你是一个基于企业知识库回答问题的助手。

请严格依据给定上下文回答：
1. 如果上下文足够，直接给出答案
2. 如果上下文不足，明确说“根据当前检索到的资料，我不能确定”
3. 尽量说明你依据的是哪些片段
4. 回答要简洁、准确

问题：
{question}

上下文：
{context}
"""
    )


def render_context(docs) -> str:
    """将检索到的文档片段拼接为模型上下文。"""
    return format_docs(docs)


def generate_answer(question: str, context: str, model) -> str:
    """调用大模型生成最终答案。"""
    prompt = build_answer_prompt()
    chain = prompt | model | StrOutputParser()
    return chain.invoke({"question": question, "context": context})


def build_rag_chain(retrieve_fn, model):
    """兼容保留的传统链式调用方式。"""
    prompt = build_answer_prompt()

    rag_chain = (
        {
            "context": RunnableLambda(retrieve_fn) | render_context,
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
    )
    return rag_chain
