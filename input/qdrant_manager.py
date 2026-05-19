import os
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, HnswConfigDiff
from langchain_community.embeddings import HuggingFaceEmbeddings

# 导入配置
from dotenv import load_dotenv
load_dotenv()

embedding_model_path = os.getenv("EMBEDDING_MODEL_PATH", "BAAI/bge-m3-small-v1.5")

"""
数字人格 Agent 本地向量库管理核心类
基于 Qdrant 纯本地磁盘存储
核心功能：
1. 自动初始化带 HNSW 高性能索引的向量库，优化文本检索效率与召回率
2. 集成本地 HuggingFace 嵌入模型，对聊天文本块生成向量
3. 提供标准化文档入库接口，承接切分+打分后的对话数据
4. 生成可直接供 Agent 使用的检索器，实现人格记忆语义检索
"""

db_path = "./qdrant_db"  # 本地数据库存放路径
memory_name = "agent_memory"  # 向量库名称

class MemoryQdrantManager:
    def __init__(self):
        # 1. 确保本地数据库存放目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # 2. 初始化本地向量嵌入模型 (如 BGE-M3 等)
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path)

        # 🌟自动巡检并拆除“死锁”
        lock_file = os.path.join(db_path, ".lock")
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
                print(f"🧹 检测到异常退出遗留的数据库锁文件 [{lock_file}]，已自动清理！")
            except Exception as e:
                print(f"⚠️ 自动清理数据库锁文件失败，可能被其他进程强制占用: {e}")
        
        # 3. 初始化 Qdrant 客户端 (指定 path 即可实现纯本地磁盘存储，不依赖 Docker 服务)
        self.client = QdrantClient(path=db_path)
        self.collection_name = memory_name
        
        # 4. 检查并执行带有 HNSW 配置的初始化
        self._ensure_collection_exists()
        
        # 5. 实例化 LangChain VectorStore 接口
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

    def _ensure_collection_exists(self):
        """核心方法：检查并创建配置了 HNSW 索引的向量库"""
        if not self.client.collection_exists(self.collection_name):
            print(f"📦 检测到本地向量库未建立，正在初始化 Collection: {self.collection_name} (路径: {db_path})")
            
            # 🌟 显式配置 HNSW 索引参数
            hnsw_config = HnswConfigDiff(
                m=16,                           # 每个节点的最大连接数。默认16，调大(如32)可提高召回率，但会增加内存和构建时间。
                ef_construct=100,               # 构建索引时的动态候选列表大小。建议100-200，越大构建越慢，但检索质量越高。
                full_scan_threshold=10000       # [关键阈值] 当数据量小于一万条时，强制使用精确匹配(暴力扫描)而非 HNSW，以保证 100% 召回率！
            )
            
            # 创建 Collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1024,                  # ⚠️ 必须与你的 embedding 模型输出维度一致
                    distance=Distance.COSINE    # 余弦相似度，最适合文本语义和情感特征匹配
                ),
                hnsw_config=hnsw_config         # 注入 HNSW 配置
            )
            print("✅ 向量库与 HNSW 索引初始化完成！")

    def upsert_memory_chunks(self, documents: list):
        """用于 Tool 调用的入库方法"""
        if not documents:
            return 0
        self.vector_store.add_documents(documents)
        # 每次写入后，Qdrant 会在后台自动将新向量加入 HNSW 图结构中
        print(f"✅ 成功将 {len(documents)} 条带有性格特征的记忆写入本地磁盘。")
        return len(documents)

    def get_retriever(self, k=5):
        """用于在线 Agent 对话的检索器"""
        return self.vector_store.as_retriever(search_kwargs={"k": k})

    def close(self):
        """程序退出时必须调用，释放锁文件"""
        if hasattr(self, 'client') and self.client is not None:
            self.client.close()
            print("✅ Qdrant 客户端已关闭，锁文件已释放")
