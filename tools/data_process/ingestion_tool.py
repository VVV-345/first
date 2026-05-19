from langchain_core.tools import tool
from tools.data_process.data_clear import TimeAwareChatLoader
from tools.data_process.scorer_tool import process_and_score_memories
from input.qdrant_manager import MemoryQdrantManager


@tool
def execute_memory_ingestion_pipeline(data_dir: str) -> str:
    """
    当用户要求清洗数据、处理聊天记录、提取性格打分或更新记忆库时，调用此工具。
    只需传入数据的目录路径（如 'input/data/'）。
    """
    print(f"\n⚙️ 流水线启动：正在读取 {data_dir}...")
    
    # 步骤 A：加载切分
    loader = TimeAwareChatLoader(data_dir)
    raw_documents = list(loader.lazy_load())
    
    # 步骤 B：远端打分 (自带防断点和防重复)
    scored_documents = process_and_score_memories(raw_documents)
    
    # 步骤 C：入库
    if scored_documents:
        db = MemoryQdrantManager()
        db.upsert_memory_chunks(scored_documents)
        return f"调度成功！共切分 {len(raw_documents)} 块，成功打分并入库 {len(scored_documents)} 块新记忆。"
    else:
        return "扫描完毕，没有发现需要打分入库的新数据。"
    


