from input.qdrant_manager import MemoryQdrantManager

def get_persona_retriever():
    """
    实例化并返回一个配置好 MMR 策略的 Qdrant 检索器
    """
    db_manager = MemoryQdrantManager()
    vector_store = db_manager.vector_store
    
    # 将底层的 Qdrant 转换为 LangChain 标准检索器
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,          # 最终喂给大模型的记忆条数
            "fetch_k": 20,   # 候选池大小
            "lambda_mult": 0.5 # 多样性控制参数
        }
    )
    return retriever