"""LLM 结构化评分引擎 — MemoryFeatures 模型 + 批量并发评分"""
import os
import hashlib
import json
import logging
from typing import List
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from config import LLM_MODEL, LLM_API_KEY, LLM_BASE_URL, CHAT_NAME
from persona.registry import load_processed_ids, REGISTRY_FILE
from persona.profile import load_global_persona, save_global_persona, update_ema

logger = logging.getLogger(__name__)


class MemoryFeatures(BaseModel):
    """基于大五人格与客观事实的特征抽取"""

    instant_emotion: str = Field(..., description="瞬时情绪：喜悦、愤怒、悲伤、平静、恐惧等")

    core_facts: list[str] = Field(
        default=[],
        description="从对话中提取的关键事实或信息点（如喜好、约定、经历）。毫无营养的废话返回空列表。"
    )

    impact_factor: float = Field(
        ...,
        ge=0.01, le=0.50,
        description="冲击力系数(0.01-0.50)。如果是吃喝拉撒等毫无营养的废话给0.01；如果是深刻的表白、剧烈的争吵、重大的生活变故，给0.3甚至0.5。"
    )

    openness: int = Field(..., ge=0, le=10)
    conscientiousness: int = Field(..., ge=0, le=10)
    extraversion: int = Field(..., ge=0, le=10)
    agreeableness: int = Field(..., ge=0, le=10)
    neuroticism: int = Field(..., ge=0, le=10)


def _build_scoring_chain():
    """构造 LLM 评分链（prompt + structured output）"""
    cloud_llm = ChatOpenAI(
        model=LLM_MODEL, api_key=LLM_API_KEY, base_url=LLM_BASE_URL,
        temperature=0.1, max_retries=3, timeout=45
    )
    structured_scorer = cloud_llm.with_structured_output(MemoryFeatures, method="json_mode")

    sys_prompt = f"""你是一个专业的心理学与语言学文本分析引擎。
你的任务是阅读一段双人聊天记录，并提取出核心信息。

⚠️ 核心规则：请务必只针对【{CHAT_NAME}】在这个对话中的表现，进行【瞬时情绪】提取和【大五人格】打分！不要对"我"（用户）的表现进行性格打分。

【核心事实 (core_facts)】
结合上下文，提取客观事实、重要约定或双方的偏好（如："用户今天带猫去打针了"、"{CHAT_NAME}表示自己不吃香菜"）。
如果是毫无信息量的废话，请务必返回空列表 []。绝对不要捏造事实！

【记忆冲击力系数 (impact_factor)】
⚠️ 评估这段对话对【{CHAT_NAME}】长期性格和情绪的冲击力（0.01 - 0.50）：
- 0.01-0.05：日常废话、打招呼。
- 0.06-0.15：普通的情绪表达或小事分享。
- 0.16-0.30：明显的争吵、深度的走心交流、建立重要约定。
- 0.31-0.50：极端重大的情感变故、极度深刻的表白。

【{CHAT_NAME}的大五人格打分标尺 (0-10分)】
1. 开放性: 聊新鲜事物、用词丰富给高分(7+)；只说"哦""好的"给低分(3-)。
2. 尽责性: 表达明确、有规划给高分(7+)；随口敷衍、多变给低分(3-)。
3. 外向性: 主动找话题、热情给高分(7+)；被动、字数极少给低分(3-)。
4. 宜人性: 表达关心、温柔、共情给高分(8+)；傲娇、攻击、冷漠、指责给低分(4-)。
5. 神经质: 抱怨、焦虑、情绪极度不稳定给高分(7+)；极度冷静、平和给低分(3-)。

⚠️ 致命警告：你必须严格输出一个【扁平的】单层 JSON 对象。绝对不允许把打分或者系数放进 core_facts 列表里！
必须严格按照以下键值对格式输出：
{{{{
    "instant_emotion": "具体情绪词",
    "core_facts": ["事实1", "事实2"],
    "impact_factor": 0.05,
    "openness": 5,
    "conscientiousness": 5,
    "extraversion": 5,
    "agreeableness": 5,
    "neuroticism": 5
}}}}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        ("human", "请提取以下聊天记录的特征：\n\n{chat_text}")
    ])

    return prompt | structured_scorer


def process_and_score_memories(documents: List[Document], start_offset: int = 0) -> List[Document]:
    """批量并发评分，含断点续传和 EMA 人格更新

    Args:
        documents: 待评分的文档列表
        start_offset: 全局偏移量，用于跨批次进度显示（如第二次调用传入 100）
    """
    if not documents:
        return []

    structured_scorer_chain = _build_scoring_chain()

    processed_ids = load_processed_ids()
    global_profile = load_global_persona()

    scored_documents = []
    BATCH_SIZE = 32
    total_in_batch = len(documents)
    logger.info("本批 %d 块（全局偏移 %d），子批次大小: %d", total_in_batch, start_offset, BATCH_SIZE)

    for i in range(0, len(documents), BATCH_SIZE):
        batch_docs = documents[i: i + BATCH_SIZE]

        docs_to_process = []
        for doc in batch_docs:
            unique_string = f"{doc.metadata.get('timestamp', '')}_{doc.page_content}"
            chunk_id = hashlib.md5(unique_string.encode('utf-8')).hexdigest()
            if chunk_id not in processed_ids:
                docs_to_process.append((doc, chunk_id))

        if not docs_to_process:
            continue

        first_global = start_offset + i + 1
        last_global = start_offset + i + len(batch_docs)
        logger.info("打分 [全局 %d-%d/%d] (子批次 %d, 新块 %d)...",
                     first_global, last_global, start_offset + total_in_batch,
                     i // BATCH_SIZE + 1, len(docs_to_process))

        batch_inputs = [{"chat_text": doc.page_content} for doc, _ in docs_to_process]

        batch_results = structured_scorer_chain.batch(
            batch_inputs,
            config={"max_concurrency": 3},
            return_exceptions=True
        )

        for (doc, chunk_id), features in zip(docs_to_process, batch_results):
            if isinstance(features, BaseException):
                logger.error("块 %s 分析失败: %s", chunk_id, features)
                continue

            # 强转回 Pydantic 实例
            if isinstance(features, dict):
                try:
                    features = MemoryFeatures(**features)
                except Exception as pydantic_err:
                    logger.error("块 %s 字典转换为 Pydantic 失败: %s, 原始数据: %s", chunk_id, pydantic_err, features)
                    continue

            if features is None:
                logger.warning("块 %s 评分返回 None（LLM 调用异常），跳过", chunk_id)
                continue

            is_empty_talk = (
                all(4 <= getattr(features, d) <= 6 for d in
                    ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"])
                and len(features.core_facts) == 0
            )

            # 每个文档分配唯一 sequence_id（不论是否空对话）
            global_profile["total_memories"] += 1
            doc.metadata["sequence_id"] = global_profile["total_memories"]

            if not is_empty_talk:
                update_ema(
                    global_profile,
                    features.openness, features.conscientiousness,
                    features.extraversion, features.agreeableness,
                    features.neuroticism, features.impact_factor
                )
            doc.metadata.update(features.model_dump())
            doc.metadata["chunk_id"] = chunk_id

            scored_documents.append(doc)
            processed_ids.add(chunk_id)

        # 每批次结束统一存档（原子写入，防崩溃损坏）
        try:
            tmp_reg = REGISTRY_FILE + ".tmp"
            with open(tmp_reg, 'w', encoding='utf-8') as f:
                json.dump(list(processed_ids), f)
            os.replace(tmp_reg, REGISTRY_FILE)
        except OSError:
            logger.error("写入断点文件失败: %s", REGISTRY_FILE)
        save_global_persona(global_profile)

        logger.info("第 %d 批次处理完毕并存档。", i // BATCH_SIZE + 1)

    return scored_documents
