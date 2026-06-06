"""LangChain @tool 封装 — 供 Agent 调用的记忆灌注工具"""
import logging
from langchain_core.tools import tool
from ingestion.loader import TimeAwareChatLoader
from scoring.scorer import process_and_score_memories
from retrieval.qdrant_store import MemoryQdrantManager

logger = logging.getLogger(__name__)


@tool
def execute_memory_ingestion_pipeline(data_dir: str) -> str:
    """
    当用户要求清洗数据、处理聊天记录、提取性格打分或更新记忆库时，调用此工具。
    只需传入数据的目录路径（如 'data/'）。
    """
    logger.info("流水线启动：正在读取 %s...", data_dir)

    loader = TimeAwareChatLoader(data_dir)

    # 分批收集并打分，避免 list() 一次性把生成器全部具象化导致 OOM
    db = MemoryQdrantManager()
    total_raw = 0
    total_scored = 0

    try:
        batch_size = 100
        raw_batch = []

        for doc in loader.lazy_load():
            raw_batch.append(doc)
            if len(raw_batch) >= batch_size:
                scored = process_and_score_memories(raw_batch, start_offset=total_raw)
                if scored:
                    db.upsert_memory_chunks(scored)
                    total_scored += len(scored)
                total_raw += len(raw_batch)
                logger.info("▸ 全局进度: 已遍历 %d 块, 新增入库 %d 块", total_raw, total_scored)
                raw_batch.clear()

        # 处理剩余不足一批的文档
        if raw_batch:
            scored = process_and_score_memories(raw_batch, start_offset=total_raw)
            if scored:
                db.upsert_memory_chunks(scored)
                total_scored += len(scored)
            total_raw += len(raw_batch)
    finally:
        # 无论成功失败，必须释放文件锁，否则 Streamlit 子进程无法打开同一数据库
        db.close()

    if total_scored > 0:
        return f"调度成功！共切分 {total_raw} 块，成功打分并入库 {total_scored} 块新记忆。"
    else:
        return "扫描完毕，没有发现需要打分入库的新数据。"
