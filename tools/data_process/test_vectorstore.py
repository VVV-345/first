import os
import json
from input.qdrant_manager import MemoryQdrantManager


def is_memory_database_exists() -> bool:
    """
    检查记忆向量数据库是否已经创建并包含数据
    返回：True=存在且有数据，False=不存在/空库
    """
    try:
        db = MemoryQdrantManager()
        client = db.client
        collection_name = db.collection_name

        # 检查集合是否存在
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if collection_name not in collection_names:
            return False
        
        # 检查数据量 > 0
        count_result = client.count(collection_name=collection_name)
        print(f"✅ 检测到已有数据库（集合: {collection_name}），数据量: {count_result} 条")
        print("⏭️  跳过加载、打分、入库步骤，直接结束流水线。")
        return count_result.count > 0  # Qdrant count 返回对象，用 .count 获取数值

    except Exception as e:
        print(f"⚠️ 数据库检查异常：{str(e)}")
        return False
    


    