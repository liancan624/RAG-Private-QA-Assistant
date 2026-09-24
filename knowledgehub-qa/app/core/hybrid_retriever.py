'''
封装双路召回 + 融合逻辑
'''

from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.db.bm25_store import BM25Store
from app.config import BM25_WEIGHT, VECTOR_WEIGHT, HYBRID_TOP_K

class HybridRetriever(BaseRetriever):
    '''混合检索器：BM25 关键词检索 + 向量语义检索，加权融合去重'''
    vector_retriever: object    # 向量检索器
    bm25_store: BM25Store       # BM25 检索器
    top_k: int = HYBRID_TOP_K

    class Config:
        arbitraty_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # 1. 两路并行召回
        vector_results = self._vector_search(query)
        bm25_results = self.bm25_store.search(query, top_k=self.top_k)

        # 2. 分数归一化 + 加权融合
        doc_scores = {}

        # 向量结果归一化（最大最小归一化）
        if vector_results:
            max_vec = max(score for _, score in vector_results) if vector_results else 1
            min_vec = min(score for _, score in vector_results) if vector_results else 0
            vec_range = max_vec - min_vec if max_vec != min_vec else 1

            for doc, score in vector_results:
                norm_score = (max_vec - score) / vec_range
                doc_id = self._doc_id(doc)
                doc_scores[doc_id] = {
                    "doc": doc,
                    "total": norm_score * VECTOR_WEIGHT
                }

        # BM25 结果归一化
        if bm25_results:
            max_bm = max(score for _, score in bm25_results) if bm25_results else 1
            min_bm = min(score for _, score in bm25_results) if bm25_results else 0
            bm_range = max_bm - min_bm if max_bm != min_bm else 1

            for doc, score in bm25_results:
                norm_score = (score - min_bm) / bm_range
                doc_id = self._doc_id(doc)
                if doc_id in doc_scores:
                    doc_scores[doc_id]["total"] += norm_score * BM25_WEIGHT
                else:
                    doc_scores[doc_id] = {
                        "doc": doc,
                        "total": norm_score * BM25_WEIGHT
                    }

        # 3. 按总分排序，取 Top K
        scored_items = sorted(doc_scores.values(), key=lambda x: x["total"], reverse=True)
        return [item["doc"] for item in scored_items[:self.top_k]]

    def _vector_search(self, query: str) -> List[tuple[Document, float]]:
        '''执行向量检索，返回带相似度分数的结果'''
        # 调用向量检索器的相似度搜索
        docs_with_score = self.vector_retriever.vectorstore.similarity_search_with_score(query, k=self.top_k)
        return docs_with_score

    @staticmethod
    def _doc_id(doc: Document) -> str:
        '''生成文档唯一标识，用于去重'''
        # 用来源+内容哈希作为唯一ID，避免同一文档被两路重复召回
        source = doc.metadata.get("source", "")
        content_hash = hash(doc.page_content[:100])
        return f"{source}_{content_hash}"










