'''
文档管理业务封装
'''

import os
from typing import List, Tuple
from fastapi import UploadFile

from app.config import KNOWLEDGE_BASE_DIR

class DocumentService:
    @staticmethod
    def init_root_dir():
        '''初始化知识库根目录，启动时自动调用，无需手动创建'''
        os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)

    @staticmethod
    def get_kb_path(kb_id: str) -> str:
        """获取知识库文件目录路径"""
        return os.path.join(KNOWLEDGE_BASE_DIR, kb_id)

    @staticmethod
    def get_docs_path(kb_id: str) -> str:
        """获取原始文档存储目录"""
        return os.path.join(DocumentService.get_kb_path(kb_id), "docs")

    @staticmethod
    def get_chroma_path(kb_id: str) -> str:
        """获取知识库向量库存储路径"""
        return os.path.join(DocumentService.get_kb_path(kb_id), "chroma_db")

    @staticmethod
    def get_bm25_path(kb_id: str) -> str:
        """获取 BM25 索引存储目录"""
        return os.path.join(DocumentService.get_kb_path(kb_id), "bm25")

    @staticmethod
    def create_kb(kb_id: str) -> dict:
        '''创建新知识库，返回创建状态'''
        DocumentService.init_root_dir()
        kb_path = DocumentService.get_kb_path(kb_id)
        if os.path.exists(kb_path):
            return {"created": False, "message": "知识库已存在"}

        # 创建子目录
        os.makedirs(DocumentService.get_docs_path(kb_id), exist_ok=True)
        os.makedirs(DocumentService.get_chroma_path(kb_id), exist_ok=True)
        os.makedirs(DocumentService.get_bm25_path(kb_id), exist_ok=True)

        return {"create": True, "message": "知识库创建成功"}

    @staticmethod
    def upload_files(kb_id: str, files: List[UploadFile]) -> Tuple[List[str], List[dict]]:
        """上传文件到指定知识库的 docs 目录"""
        docs_dir = DocumentService.get_docs_path(kb_id)
        os.makedirs(docs_dir, exist_ok=True)

        saved_files = []
        failed_files = []

        for file in files:
            try:
                file_path = os.path.join(docs_dir, file.filename)
                with open(file_path, "wb") as f:
                    f.write(file.file.read())
                saved_files.append(file.filename)
            except Exception as e:
                failed_files.append({"filename": file.filename, "error": str(e)})

        return saved_files, failed_files

    @staticmethod
    def list_kb() -> List[dict]:
        """获取所有知识库列表与状态"""
        DocumentService.init_root_dir()
        kb_list = []
        if not os.path.exists(KNOWLEDGE_BASE_DIR):
            return kb_list

        for name in os.listdir(KNOWLEDGE_BASE_DIR):
            kb_path = DocumentService.get_kb_path(name)
            if not os.path.isdir(kb_path):
                continue

            docs_path = DocumentService.get_docs_path(name)
            file_count = 0
            if os.path.exists(docs_path):
                file_count = len([
                    f for f in os.listdir(docs_path)
                    if os.path.isfile(os.path.join(docs_path, f))
                ])

            has_db = os.path.exists(DocumentService.get_chroma_path(name))

            kb_list.append({
                "kb_id": name,
                "file_count": file_count,
                "built": has_db
            })

        return kb_list
