'''
全局配置统一管理，所有常量、路径、默认参数集中在此
'''

from pathlib import Path

# ========== 路径配置 ==========
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_bases"

# ========== 模型配置 ==========
EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
EMBED_DEVICE = "cpu"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANKER_DEVICE = "cpu"

# ========== 检索默认参数 ==========
DEFAULT_RECALL_TOP_K = 10
DEFAULT_RERANK_TOP_K = 3
DEFAULT_CHUNK_THRESHOLD = 85

# ========== 服务配置 ==========
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000

# ========== 混合检索配置 ==========
BM25_WEIGHT = 0.4       # BM25 关键词检索权重
VECTOR_WEIGHT = 0.6     # 向量语义检索权重
HYBRID_TOP_K = 20       # 混合召回候选数量
