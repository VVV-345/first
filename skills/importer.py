"""技能导入器 — 自动识别并转换社区格式为标准 .skill.json 结构"""
import json
import logging
import os

logger = logging.getLogger(__name__)


def detect_and_import(file_bytes: bytes, filename: str) -> dict | None:
    """
    根据文件名和内容自动识别格式，返回标准 skill dict。
    支持格式:
      - .skill.json   → 直接校验
      - Character Card v2 .json  → 提取角色设定
      - .txt / .md    → 全文作为 prompt_injection
    """
    ext = os.path.splitext(filename)[1].lower()

    # 1. 原生 .skill.json
    if filename.endswith(".skill.json"):
        from skills.loader import validate_skill
        try:
            data = json.loads(file_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error("无法解析技能文件: %s", e)
            return None
        if validate_skill(data):
            data.setdefault("version", "1.0")
            data.setdefault("description", "")
            data.setdefault("icon", "🧩")
            return data
        return None

    # 2. JSON → 尝试 Character Card v2
    if ext == ".json":
        try:
            data = json.loads(file_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.error("无法解析 JSON 文件")
            return None
        if "spec" in data and "data" in data:
            return _import_chara_card(data)
        logger.warning("JSON 文件不是 .skill.json 也不是 Character Card v2，无法识别")
        return None

    # 3. 纯文本 .txt / .md
    if ext in (".txt", ".md"):
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            logger.error("无法解码文本文件: %s", e)
            return None
        name = os.path.splitext(filename)[0]
        return _import_plain_text(text, name)

    logger.warning("不支持的文件类型: %s", ext)
    return None


def _import_chara_card(data: dict) -> dict:
    """从 Character Card v2 格式提取角色设定"""
    spec_data = data.get("data", {})
    name = spec_data.get("name", "未命名角色")
    description = spec_data.get("description", "")
    personality = spec_data.get("personality", "")
    scenario = spec_data.get("scenario", "")
    first_mes = spec_data.get("first_mes", "")

    prompt_parts = []
    if description:
        prompt_parts.append(f"【角色设定】{description}")
    if personality:
        prompt_parts.append(f"【性格】{personality}")
    if scenario:
        prompt_parts.append(f"【场景】{scenario}")
    if first_mes:
        prompt_parts.append(f"【首条消息参考】{first_mes}")

    prompt_injection = "\n".join(prompt_parts) if prompt_parts else description

    return {
        "name": name,
        "version": "1.0",
        "description": description[:100] if description else "",
        "icon": "🐱",
        "source": "chara_card_v2",
        "prompt_injection": prompt_injection,
    }


def _import_plain_text(text: str, name: str) -> dict:
    """从纯文本文件导入技能"""
    desc = text[:100].replace("\n", " ") + ("..." if len(text) > 100 else "")
    return {
        "name": name,
        "version": "1.0",
        "description": desc,
        "icon": "📄",
        "source": "plain_text",
        "prompt_injection": text.strip(),
    }
