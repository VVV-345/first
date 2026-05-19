import os
import json
import hashlib
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import FileCallbackHandler

from dotenv import load_dotenv
# 导入process_model,apikey,base_url
load_dotenv()
process_model = os.getenv("PROCESS_MODEL")
llm_api_key = os.getenv("LLM_API_KEY")
base_url = os.getenv("BASE_URL") 
CHAT_NAME = os.getenv("CHAT_NAME") 
# ==========================================
# 1. 断点续传与全局人格的文件路径
# ==========================================
REGISTRY_FILE = "input/processed_registry.json"
PERSONA_FILE = "input/global_persona_config.json" # 全局画像配置文件

def load_processed_ids() -> set:
    """从本地文件加载已处理过的文本块 ID 集合"""
    if os.path.exists(REGISTRY_FILE):
        try:
            with open(REGISTRY_FILE, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        except Exception as e:
            print(f"⚠️ 读取进度文件失败: {e}")
    return set()

def save_processed_id(chunk_id: str):
    """将新处理完成的 ID 实时追加到本地注册表中"""
    processed = load_processed_ids()
    processed.add(chunk_id)
    # 实时写入，确保即使程序崩溃也能保存最新进度
    with open(REGISTRY_FILE, 'w', encoding='utf-8') as f:
        json.dump(list(processed), f)

# ---------------- 全局人格逻辑  ----------------
def load_global_persona() -> dict:
    """检查文件是否存在，存在则读取，不存在则返回初始值 5.0"""
    if os.path.exists(PERSONA_FILE):
        try:
            with open(PERSONA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ 读取全局画像失败，将使用初始值: {e}")
            
    # 初始默认值
    return {
        "total_memories": 0,
        "avg_openness": 5.0,
        "avg_conscientiousness": 5.0,
        "avg_extraversion": 5.0,
        "avg_agreeableness": 5.0,
        "avg_neuroticism": 5.0
    }

def save_global_persona(profile: dict):
    """将最新的全局画像覆盖写入文件"""
    with open(PERSONA_FILE, 'w', encoding='utf-8') as f:
        json.dump(profile, f, ensure_ascii=False, indent=4)


# ==========================================
# 2. 定义大语言模型必须遵守的输出结构
# ==========================================
class MemoryFeatures(BaseModel):
    """基于大五人格与客观事实的特征抽取"""
    
    instant_emotion: str = Field(..., description="瞬时情绪：喜悦、愤怒、悲伤、平静、恐惧等")
    
    # 🌟 事实提取器
    core_facts: list[str] = Field(
        default=[], 
        description="从对话中提取的关键事实或信息点（如喜好、约定、经历）。毫无营养的废话返回空列表。"
    )

    # 🌟 冲击力系数
    impact_factor: float = Field(
        ..., 
        ge=0.01, le=0.50, 
        description="冲击力系数(0.01-0.50)。如果是吃喝拉撒等毫无营养的废话给0.01；如果是深刻的表白、剧烈的争吵、重大的生活变故，给0.3甚至0.5。"
    )
    
    # 🌟 五大人格维度
    openness: int = Field(..., ge=0, le=10)
    conscientiousness: int = Field(..., ge=0, le=10)
    extraversion: int = Field(..., ge=0, le=10)
    agreeableness: int = Field(..., ge=0, le=10)
    neuroticism: int = Field(..., ge=0, le=10)

# ==========================================
# 3. 封装打分
# ==========================================
def process_and_score_memories(documents: List[Document], callbacks=None) -> List[Document]:
    if not documents:
        return []

    if not callbacks:
        callbacks = [FileCallbackHandler("ingestion_trace.log")]

    cloud_llm = ChatOpenAI(
        model=process_model, api_key=llm_api_key, base_url=base_url,
        temperature=0.1, max_retries=3, timeout=45,  callbacks=callbacks      
    )
    structured_scorer = cloud_llm.with_structured_output(MemoryFeatures)
    
    sys_prompt = f"""你是一个专业的心理学与语言学文本分析引擎。
    你的任务是阅读一段双人聊天记录，并提取出核心信息。
    
    ⚠️ 核心规则：请务必只针对【{CHAT_NAME}】在这个对话中的表现，进行【瞬时情绪】提取和【大五人格】打分！不要对“我”（用户）的表现进行性格打分。

    【核心事实 (core_facts)】
    结合上下文，提取客观事实、重要约定或双方的偏好（如：“用户今天带猫去打针了”、“{CHAT_NAME}表示自己不吃香菜”）。
    如果是毫无信息量的废话，请务必返回空列表 []。绝对不要捏造事实！

    【记忆冲击力系数 (impact_factor)】
    ⚠️ 评估这段对话对【{CHAT_NAME}】长期性格和情绪的冲击力（0.01 - 0.50）：
    - 0.01-0.05：日常废话、打招呼。
    - 0.06-0.15：普通的情绪表达或小事分享。
    - 0.16-0.30：明显的争吵、深度的走心交流、建立重要约定。
    - 0.31-0.50：极端重大的情感变故、极度深刻的表白。

    【{CHAT_NAME}的大五人格打分标尺 (0-10分)】
    1. 开放性: 聊新鲜事物、用词丰富给高分(7+)；只说“哦”“好的”给低分(3-)。
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

    # 构造Prompt模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        ("human", "请提取以下聊天记录的特征：\n\n{chat_text}")
    ])

    # 把Prompt和结构化输出的LLM绑定
    structured_scorer_chain = prompt | structured_scorer


    # 🌟 加载双状态：已处理 ID 和 当前的全局人格
    processed_ids = load_processed_ids()
    global_profile = load_global_persona()
    
    scored_documents = []

    BATCH_SIZE = 32
    print(f"\n📦 开始处理，总计 {len(documents)} 个数据块，批次大小: {BATCH_SIZE}")

    # 将所有文档按 BATCH_SIZE 进行切片循环
    for i in range(0, len(documents), BATCH_SIZE):
        batch_docs = documents[i : i + BATCH_SIZE]
        
        # 筛选出这个批次中真正需要处理的新块
        docs_to_process = []
        for doc in batch_docs:
            unique_string = f"{doc.metadata.get('timestamp', '')}_{doc.page_content}"
            chunk_id = hashlib.md5(unique_string.encode('utf-8')).hexdigest()
            if chunk_id not in processed_ids:
                docs_to_process.append((doc, chunk_id))
                
        if not docs_to_process:
            continue # 如果这一批 32 个全都在之前处理过了，直接跳过
            
        print(f"\n🔄 正在并发打分第 {i//BATCH_SIZE + 1} 批次 (包含 {len(docs_to_process)} 个新数据块)...")
        
        # 准备批量发给大模型的输入
        batch_inputs = [{"chat_text": doc.page_content} for doc, _ in docs_to_process]
        
        # 🌟 开启批量并发请求！
        # max_concurrency 设定为 3，表示最高允许同时发起 3 个网络请求，避免被封 IP
        # return_exceptions=True 极其重要：如果某个块彻底失败，不会导致整个批次崩溃，而是返回 Exception 对象
        batch_results = structured_scorer_chain.batch(
            batch_inputs, 
            config={"max_concurrency": 3,
                    "callbacks": callbacks
                    }, 
            return_exceptions=True
        )
        
        # 拿到并发结果后，按时间顺序依次处理
        for (doc, chunk_id), features in zip(docs_to_process, batch_results):
            # 如果打分失败
            if isinstance(features, Exception):
                print(f"❌ 块 {chunk_id} 分析失败: {features}")
                continue # 跳过这个块，下次重新运行会自动重试它
                
            # 如果打分成功，执行 EMA 算法更新
            is_empty_talk = (
                features.openness == 5 and features.conscientiousness == 5 and 
                features.extraversion == 5 and features.agreeableness == 5 and 
                features.neuroticism == 5 and len(features.core_facts) == 0
            )
            
            if not is_empty_talk:
                alpha = max(0.01, min(0.25, features.impact_factor * 0.4))
                global_profile["avg_openness"] = (alpha * features.openness) + ((1 - alpha) * global_profile["avg_openness"])
                global_profile["avg_conscientiousness"] = (alpha * features.conscientiousness) + ((1 - alpha) * global_profile["avg_conscientiousness"])
                global_profile["avg_extraversion"] = (alpha * features.extraversion) + ((1 - alpha) * global_profile["avg_extraversion"])
                global_profile["avg_agreeableness"] = (alpha * features.agreeableness) + ((1 - alpha) * global_profile["avg_agreeableness"])
                global_profile["avg_neuroticism"] = (alpha * features.neuroticism) + ((1 - alpha) * global_profile["avg_neuroticism"])
                global_profile["total_memories"] += 1
            
            doc.metadata["sequence_id"] = global_profile["total_memories"]
            doc.metadata.update(features.dict())
            doc.metadata["chunk_id"] = chunk_id
            
            scored_documents.append(doc)
            processed_ids.add(chunk_id) # 记录到内存中
            
        # 🌟 每一个 Batch 处理完之后，统一保存一次本地进度
        with open(REGISTRY_FILE, 'w', encoding='utf-8') as f:
            json.dump(list(processed_ids), f)
        save_global_persona(global_profile)
        
        print(f"✅ 第 {i//BATCH_SIZE + 1} 批次处理完毕并存档。")

    return scored_documents   
