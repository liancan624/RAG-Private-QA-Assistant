'''
向量库业务封装，带缓存机制
'''

import os

from app.config import (
    DEFAULT_CHUNK_THRESHOLD,
    EMBED_MODEL_NAME,
    EMBED_DEVICE,
    HYBRID_TOP_K
)
from app.core.models import embedding_model
from app.utils.doc_processor import load_folder_documents, semantic_chunk_documents
from app.db.vector_db import build_vector_store, load_vector_store, get_retriever
from app.db.bm25_store import BM25Store
from app.core.hybrid_retriever import HybridRetriever
from app.services.document_service import DocumentService

class VectorService:
    _kb_cache = {}      # 向量库实例缓存
    _bm25_cache = {}    # BM25 索引缓存

    @classmethod
    def build_kb(cls, kb_id: str) -> dict:
        """构建指定知识库的向量索引"""
        docs_dir = DocumentService.get_docs_path(kb_id)

        documents = load_folder_documents(docs_dir)
        if not documents:
            raise ValueError("该知识库下没有可加载的文档")

        split_docs = semantic_chunk_documents(
            documents,
            EMBED_MODEL_NAME,
            EMBED_DEVICE,
            threshold=DEFAULT_CHUNK_THRESHOLD
        )

        # 1. 构建向量库
        chroma_path = DocumentService.get_chroma_path(kb_id)
        vector_db = build_vector_store(split_docs, embedding_model, chroma_path)

        # 2. 构建 BM25 索引并持久化
        bm25_file = os.path.join(DocumentService.get_bm25_path(kb_id), "bm25_index.pkl")
        bm25_store = BM25Store(split_docs)
        bm25_store.save(bm25_file)

        # 3. 写入缓存
        cls._kb_cache[kb_id] = vector_db
        cls._bm25_cache[kb_id] = bm25_store

        return {"doc_count": len(documents), "chunk_count": len(split_docs)}

    @classmethod
    def get_vector_db(cls, kb_id: str):
        """获取向量库实例，优先走缓存"""
        if kb_id in cls._kb_cache:
            return cls._kb_cache[kb_id]

        chroma_path = DocumentService.get_chroma_path(kb_id)
        if not os.path.exists(chroma_path):
            raise ValueError("知识库不存在，请先构建")

        vector_db = load_vector_store(chroma_path, embedding_model)
        cls._kb_cache[kb_id] = vector_db
        return vector_db

    @classmethod
    def get_bm25_store(cls, kb_id: str) -> BM25Store:
        '''获取 BM25 索引实例'''
        if kb_id in cls._bm25_cache:
            return cls._bm25_cache[kb_id]

        bm25_file = os.path.join(DocumentService.get_bm25_path(kb_id), "bm25_index.pkl")
        if not os.path.exists(bm25_file):
            raise ValueError("BM25 索引不存在，请重新构建知识库")

        bm25_store = BM25Store.load(bm25_file)
        cls._bm25_cache[kb_id] = bm25_store
        return bm25_store

    @classmethod
    def get_retriever(cls, kb_id: str, top_k: int = 10):
        """获取混合检索器（默认启用双路召回）"""
        vector_db = cls.get_vector_db(kb_id)
        bm25_store = cls.get_bm25_store(kb_id)

        # 向量基础检索器
        vector_retriever = vector_db.as_retriever(search_kwargs={"k": top_k})

        # 封装为混合检索器
        return HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_store=bm25_store,
            top_k=top_k
        )
