'''
    RAG 问答编排逻辑
'''

import json
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate

from app.core.reranker import BgeReranker

def init_llm(model_name: str, api_key: str, base_url: str) -> ChatOpenAI:
    """初始化大模型实例"""
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0
    )

def rewrite_query(llm: ChatOpenAI, original_query: str) -> List[str]:
    """利用大模型将用户问题改写为多个不同角度的检索词"""
    prompt_str = """
    你是一个专业的知识库检索词优化专家。
    请从不同角度、使用相关同义词或专业术语，将用户的原始问题改写为 3 个不同的检索短语，以提高在数据库中的命中率。
    请严格仅输出一个 JSON 数组格式，不要包含任何其他解释性文字，也不要带有 Markdown 代码块符号（如 ```json）。
    示例输出: ["改写词1", "改写词2", "改写词3"]

    原始问题：{question}
    """

    prompt = PromptTemplate.from_template(prompt_str)

    chain = prompt | llm.bind(temperature=0.3)

    try:
        response = chain.invoke({"question": original_query}).content
        clean_text = response.strip().removeprefix("```json").removeprefix("```").strip()
        queries = json.loads(clean_text)
        if isinstance(queries, list):
            return queries
    except Exception as e:
        print(f"Query rewrite failed: {str(e)}")

    return []

def rag_answer(
        retriever,
        llm: ChatOpenAI,
        question: str,
        reranker: Optional[BgeReranker] = None,
        rerank_top_k: int = 3
) -> Tuple[List[Document], str]:

    expanded_queries = [question]
    rewritten_queries = rewrite_query(llm, question)
    if rewritten_queries:
        expanded_queries.extend(rewritten_queries)
        print(f"扩展检索词：{expanded_queries}")

    all_docs = []
    seen_docs = set()

    for q in expanded_queries:
        docs = retriever.invoke(q)
        for doc in docs:
            doc_hash = hash(doc.page_content)
            if doc_hash not in seen_docs:
                seen_docs.add(doc_hash)
                all_docs.append(doc)

    if reranker and all_docs:
        docs = reranker.rerank(question, all_docs, top_k=rerank_top_k, threshold=0.0)
    else:
        docs = all_docs[:rerank_top_k]

    if not docs:
        return [], "抱歉，当前知识库中未找到与该问题相关的内容。"

    prompt_str = """
    你是一个专业的专属知识库问答助手。请严格遵守以下核心规则：
    1. 必须且只能基于下方的【参考内容】来回答用户的问题。
    2. 绝对禁止使用你自己的内部知识去编造、补充答案。
    3. 如果【参考内容】无法回答问题，必须直接回复：“知识库中暂无相关信息。
    
    【参考内容】：
    {context}
    
    【用户问题】：
    {question}
    """

    prompt = PromptTemplate.from_template(prompt_str)

    chain = load_qa_chain(
        llm=llm,
        chain_type="stuff",
        prompt=prompt
    )

    result = chain.invoke({"input_documents": docs, "question": question})
    answer = result["output_text"]

    if "暂无相关信息" in answer or "未找到与该问题相关的内容" in answer:
        docs = []

    return docs, answer
