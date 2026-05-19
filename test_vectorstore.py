from input.qdrant_manager import MemoryQdrantManager

def check_qdrant():
    print("========================================")
    print("🔍 Qdrant 数据库本地查看器")
    print("========================================\n")

    # 1. 实例化你的管理器，连接到数据库
    db = MemoryQdrantManager()
    
    # 提取底层的客户端和集合名称
    client = db.client
    collection_name = db.collection_name
    
    # ------------------------------------------------
    # 解决你的问题 1：怎么查看一共有多少条数据？
    # ------------------------------------------------
    count_result = client.count(collection_name=collection_name)
    print(f"📦 数据库中当前总共有: 【 {count_result.count} 】 条数据块！\n")

    # ------------------------------------------------
    # 解决你的问题 2：查看前 5 条数据
    # ------------------------------------------------
    print("👀 正在展示前 5 条数据详情...\n")
    
    # scroll 方法用于粗略浏览数据，性能极高
    records, next_page_offset = client.scroll(
        collection_name=collection_name,
        limit=5,
        with_payload=True,  # 必须为 True，把你的大五人格、情绪等数据拿出来
        with_vectors=False  # 设为 False，屏蔽掉几千维的浮点数向量，保持终端清爽
    )

    for i, record in enumerate(records):
        print(f"▼ 第 {i+1} 条记忆 (ID: {record.id})")
        
        # Qdrant 把所有的 metadata 和文本都存在 payload 这个字典里
        payload = record.payload
        metadata = payload.get("metadata", {}) 
        
        # 打印你关心的几个核心字段
        print(f"  📝 文本内容: {payload.get('page_content', '无内容')}")
        print(f"  🎭 瞬时情绪: {metadata.get('instant_emotion', '未知')}")
        print(f"  💡 核心事实: {metadata.get('core_facts', [])}")
        print(f"  💥 冲击力: {metadata.get('impact_factor', 0)}")
        print(f"  📊 性格切片: O:{metadata.get('openness')}, C:{metadata.get('conscientiousness')}, "
              f"E:{metadata.get('extraversion')}, A:{metadata.get('agreeableness')}, N:{metadata.get('neuroticism')}")
        print("-" * 50)

if __name__ == "__main__":
    check_qdrant()