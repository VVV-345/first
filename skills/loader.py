"""技能加载器 — 校验并加载 .skill.json 文件"""
import json
import logging

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {"name", "prompt_injection"}


def validate_skill(data: dict) -> bool:
    """校验技能 dict 是否包含必填字段"""
    if not isinstance(data, dict):
        return False
    missing = REQUIRED_FIELDS - set(data.keys())
    if missing:
        logger.warning("技能缺少必填字段: %s", missing)
        return False
    return True


def load_skill(filepath: str) -> dict | None:
    """从 .skill.json 文件加载并校验技能"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("无法读取技能文件 %s: %s", filepath, e)
        return None

    if not validate_skill(data):
        logger.error("技能文件 %s 校验失败", filepath)
        return None

    # 补齐可选字段默认值
    data.setdefault("version", "1.0")
    data.setdefault("description", "")
    data.setdefault("icon", "🧩")
    return data
