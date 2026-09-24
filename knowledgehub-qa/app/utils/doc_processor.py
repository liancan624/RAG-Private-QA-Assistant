'''
    文档加载与语义分块工具
'''

import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

def _get_loader(file_path: str):
    """根据文件后缀选择加载器"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".pdf", ".docx", ".doc"]:
        return UnstructuredFileLoader(
            file_path,
            mode="elements",    # 将文档拆分为独立的元素
            strategy="hi_res",  # 高分辨率策略，支持复杂的表格和OCR
            pdf_infer_table_structure=True  # 开启PDF表格结构推断
        )
    elif ext in [".txt", ".md"]:
        from langchain_community.document_loaders import TextLoader
        return TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"不支持的文件格式：{ext}")

def load_folder_documents(folder_path: str) -> List[Document]:
    """加载文件夹下所有支持格式的文档"""
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue
        try:
            loader = _get_loader(file_path)
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = filename
                if doc.metadata.get("category") == "Table":
                    doc.metadata["is_table"] = True

            documents.extend(docs)
            print(f"加载加载文件：{filename}")
        except Exception as e:
            print(f"加载文件失败 {filename}，错误：{str(e)}")
    return documents

def semantic_chunk_documents(
    documents: List[Document],
    embed_model_name: str,
    device: str,
    threshold: int = 85
) -> List[Document]:
    """语义分块"""
    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model_name,
        model_kwargs={"device": device}
    )
    text_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=threshold,
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"文档切分完成，共{len(split_docs)}个文本块")
    return split_docs
