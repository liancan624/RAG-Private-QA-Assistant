'''
实现 BM25 索引的构建、持久化、加载、检索
'''

import os
import pickle
from typing import List
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from app.utils.tokenizer import chinese_tokenizer

class BM25Store:
    def __init__(self, docs: List[Document] = None):
        self.docs: List[Document] = []
        self.bm25: BM25Okapi = None
        if docs:
            self.build_index(docs)

    def build_index(self, docs: List[Document]):
        '''构建 BM25 索引'''
        self.docs = docs
        # 对所有文档内容分词
        tokenized_corpus = [chinese_tokenizer(doc.page_content) for doc in docs]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def save(self, persist_path: str):
        '''持久化索引到本地'''
        os.makedirs(os.path.dirname(persist_path), exist_ok=True)
        with open(persist_path, "wb") as f:
            pickle.dump({"docs": self.docs, "bm25": self.bm25}, f)

    @classmethod
    def load(cls, persist_path: str) -> "BM25Store":
        '''加载本地已构建的索引'''
        if not os.path.exists(persist_path):
            raise FileNotFoundError(f"BM25 索引不存在：{persist_path}")
        with open(persist_path, "rb") as f:
            data = pickle.load(f)
        store = cls()
        store.docs = data["docs"]
        store.bm25 = data["bm25"]
        return store

    def search(self, query: str, top_k: int = 10) -> List[tuple[Document, float]]:
        '''
        BM25 检索，返回（文档，原始分数）列表
        分数越高相关性越强
        '''
        if not self.bm25:
            return []

        query_tokens = chinese_tokenizer(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        # 按分数降序排列
        scored_docs = list(zip(self.docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return scored_docs[:top_k]







