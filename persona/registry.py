"""断点续传 — 已处理文本块的 MD5 注册表管理"""
import os
import json
import logging
from config import DATA_DIR

REGISTRY_FILE = os.path.join(DATA_DIR, "processed_registry.json")
logger = logging.getLogger(__name__)


def load_processed_ids() -> set:
    """从本地文件加载已处理过的文本块 ID 集合"""
    if os.path.exists(REGISTRY_FILE):
        try:
            with open(REGISTRY_FILE, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        except Exception as e:
            logger.warning("读取进度文件失败: %s", e)
    return set()
