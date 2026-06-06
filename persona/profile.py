"""全局人格画像 — EMA 更新算法 + JSON 持久化"""
import os
import json
import logging
from config import DATA_DIR

PERSONA_FILE = os.path.join(DATA_DIR, "global_persona_config.json")
logger = logging.getLogger(__name__)

DEFAULT_RELATIONSHIP = "无话不谈的朋友"
DEFAULT_NICKNAME = "不叫名字，直接说'你'"


def load_global_persona() -> dict:
    """检查文件是否存在，存在则读取，不存在则返回初始值 5.0"""
    if os.path.exists(PERSONA_FILE):
        try:
            with open(PERSONA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning("读取全局画像失败，将使用初始值: %s", e)

    return {
        "total_memories": 0,
        "avg_openness": 5.0,
        "avg_conscientiousness": 5.0,
        "avg_extraversion": 5.0,
        "avg_agreeableness": 5.0,
        "avg_neuroticism": 5.0,
        "relationship_status": DEFAULT_RELATIONSHIP,
        "nickname_rules": DEFAULT_NICKNAME
    }


def save_global_persona(profile: dict):
    """将最新的全局画像原子写入文件（写临时文件 + rename，防崩溃损坏）"""
    tmp_file = PERSONA_FILE + ".tmp"
    with open(tmp_file, 'w', encoding='utf-8') as f:
        json.dump(profile, f, ensure_ascii=False, indent=4)
    os.replace(tmp_file, PERSONA_FILE)


def update_ema(profile: dict, openness, conscientiousness, extraversion,
               agreeableness, neuroticism, impact_factor) -> dict:
    """EMA 算法：用冲击力系数作为学习率，增量更新大五人格维度"""
    alpha = max(0.01, min(0.25, impact_factor * 0.4))
    profile["avg_openness"] = (alpha * openness) + ((1 - alpha) * profile["avg_openness"])
    profile["avg_conscientiousness"] = (alpha * conscientiousness) + ((1 - alpha) * profile["avg_conscientiousness"])
    profile["avg_extraversion"] = (alpha * extraversion) + ((1 - alpha) * profile["avg_extraversion"])
    profile["avg_agreeableness"] = (alpha * agreeableness) + ((1 - alpha) * profile["avg_agreeableness"])
    profile["avg_neuroticism"] = (alpha * neuroticism) + ((1 - alpha) * profile["avg_neuroticism"])
    return profile
