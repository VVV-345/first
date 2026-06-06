"""Qdrant 本地向量库管理 — HNSW 索引 + HuggingFace 嵌入"""
import os
import atexit
import logging
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, HnswConfigDiff
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL_PATH

logger = logging.getLogger(__name__)

DB_PATH = "./qdrant_db"
COLLECTION_NAME = "agent_memory"


class MemoryQdrantManager:
    """
    数字人格 Agent 本地向量库管理核心类
    基于 Qdrant 纯本地磁盘存储
    """
    def __init__(self):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)

        # 自动巡检并拆除"死锁"
        lock_file = os.path.join(DB_PATH, ".lock")
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
                logger.warning("检测到异常退出遗留的数据库锁文件 [%s]，已自动清理", lock_file)
            except Exception as e:
                logger.warning("自动清理数据库锁文件失败，可能被其他进程强制占用: %s", e)

        self._closed = False
        self.client = QdrantClient(path=DB_PATH)
        self.collection_name = COLLECTION_NAME
        atexit.register(self.close)

        self._ensure_collection_exists()

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

    def _ensure_collection_exists(self):
        """核心方法：检查并创建配置了 HNSW 索引的向量库"""
        if not self.client.collection_exists(self.collection_name):
            logger.info("检测到本地向量库未建立，正在初始化 Collection: %s (路径: %s)", self.collection_name, DB_PATH)

            hnsw_config = HnswConfigDiff(
                m=16,
                ef_construct=100,
                full_scan_threshold=10000
            )

            # 从本地嵌入模型动态获取维度，避免换模型后不匹配
            test_vec = self.embeddings.embed_query("dim_check")
            dim = len(test_vec)

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE
                ),
                hnsw_config=hnsw_config
            )
            logger.info("向量库与 HNSW 索引初始化完成")

    def upsert_memory_chunks(self, documents: list):
        """用于 Tool 调用的入库方法"""
        if not documents:
            return 0
        self.vector_store.add_documents(documents)
        logger.info("成功将 %d 条带有性格特征的记忆写入本地磁盘", len(documents))
        return len(documents)

    def get_retriever(self, k=5):
        """用于在线 Agent 对话的检索器"""
        return self.vector_store.as_retriever(search_kwargs={"k": k})

    def close(self):
        """程序退出时必须调用，释放锁文件（幂等，可重复调用）"""
        if self._closed:
            return
        if hasattr(self, 'client') and self.client is not None:
            self.client.close()
            self._closed = True
            try:
                atexit.unregister(self.close)
            except Exception:
                pass  # 未注册时忽略
            logger.info("Qdrant 客户端已关闭，锁文件已释放")
