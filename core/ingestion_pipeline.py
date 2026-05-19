# 文件：core/ingestion_pipeline.py

from tools.data_process.data_clear import TimeAwareChatLoader
from tools.data_process.scorer_tool import process_and_score_memories
from input.qdrant_manager import MemoryQdrantManager

class DirectIngestionPipeline:
    """
    纯代码版记忆灌注流水线。
    无 Agent 介入，直接硬编码执行切分、打分、入库流程。
    """
    def __init__(self, data_directory: str = "input/data/"):
        self.data_directory = data_directory
        self.db_manager = MemoryQdrantManager()

    def run(self, callbacks=None):
        print(f"🚀 [Direct Pipeline] 启动纯代码灌库，扫描目录: {self.data_directory}")
        
        # 1. 加载数据
        loader = TimeAwareChatLoader(directory_path=self.data_directory)
        raw_documents = list(loader.lazy_load())
        if not raw_documents:
            print("⏭️ 未发现新数据，流水线结束。")
            return False

        # 2. 远端打分 
        print(f"☁️ [Direct Pipeline] 正在为 {len(raw_documents)} 个数据块进行性格打分...")
        scored_documents = process_and_score_memories(raw_documents, callbacks=callbacks)
        if not scored_documents:
            print("⏭️ 所有数据块均已处理过或打分失败，流水线结束。")
            return False

        # 3. 入库
        print("📦 [Direct Pipeline] 正在将数据写入 Qdrant 向量库...")
        success_count = self.db_manager.upsert_memory_chunks(scored_documents)
        print(f"✅ [Direct Pipeline] 成功入库 {success_count} 条记忆！")
        self.db_manager.client.close()  # 关闭数据库连接
        return True

