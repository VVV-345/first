import os
import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage 
# 下面是核心的聊天引擎模块，负责处理用户输入，动态构建 Prompt，调用 LLM，并管理聊天历史和长期记忆更新
from tools.retriever_tool.ensemble_retriever import get_hybrid_retriever
from tools.data_process.scorer_tool import load_global_persona, process_and_score_memories
from langchain_core.documents import Document

from dotenv import load_dotenv
load_dotenv()


class PersonaEngine:
    def __init__(self):
        # 加载全局人格画像和检索工具
        self.persona = load_global_persona()
        self.retriever = get_hybrid_retriever()
        # 将人格画像的数值特征映射成文字描述，供后续 Prompt 使用
        self.persona_desc = self._map_scores_to_text(self.persona)
        # 初始化 LLM，开启流式输出
        self.llm = ChatOpenAI(
            model=os.getenv("PROCESS_MODEL"),
            api_key=os.getenv("LLM_API_KEY", "EMPTY"),
            base_url=os.getenv("BASE_URL"),
            streaming=True, 
            temperature=0.7 # 让事实提取更精准
        )
        # 初始化聊天历史，用于提供上下文记忆，后续可以改成更复杂的记忆管理模块
        self.chat_history = [] 

    def _map_scores_to_text(self, p: dict) -> str:
        desc = []
        if p.get('avg_openness', 5) > 6: desc.append("思维活跃，乐于接受新事物和新观点。")
        else: desc.append("性格偏保守，喜欢熟悉的人和事。")
        
        if p.get('avg_extraversion', 5) > 6: desc.append("外向开朗，喜欢表达，情绪容易外露。")
        else: desc.append("性格内敛，话不太多，喜欢安静。")
        
        if p.get('avg_agreeableness', 5) > 6: desc.append("温柔体贴，善解人意，说话顾及他人感受。")
        else: desc.append("说话比较直，有时带点小脾气或傲娇。")
        
        return " ".join(desc)

    def get_dynamic_prompt(self):
        """
        动态 Prompt：加入了 {dominant_emotion} 和 {core_facts} 占位符
        """
        system_template = """
你现在正在扮演一个特定的数字人格。你必须完全沉浸在角色中，绝不能承认自己是AI。

【你的基础性格底色】
{persona_desc}

【系统动态注入：心理状态】
根据触景生情（检索到的记忆），你当前的瞬时情绪基调是：[{dominant_emotion}]
请你在此次回复中，带上这种情绪色彩（比如悲伤时用词低沉，愤怒时句子简短，开心时多用语气词）。

【系统动态注入：核心事实】
以下是你和对方共同经历过的客观事实（不可违背）：
{core_facts}

【相关历史对话记录 (Context)】
以下是你突然回忆起的片段（带有当时的冲击力评分和情绪）：
{context}

【你的回复要求】
1. 语气必须极度自然、口语化，像在微信聊天。
2. 结合“当前的心理状态”和“回忆起的片段”来回应用户的最新消息。
3. 如果相关记忆的冲击力很高，表现出对此事印象深刻；如果冲击力低，表现得轻描淡写。
"""

        return ChatPromptTemplate.from_messages([
            ("system", system_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

    async def stream_chat(self, query: str):
        # 1. 检索向量库
        docs = self.retriever.invoke(query)
        
        # 🌟 2. 动态元数据解析核心逻辑
        context_parts = []
        all_facts = []
        max_impact = 0
        dominant_emotion = "平静" # 默认情绪
        
        for d in docs:
            meta = d.metadata
            # 安全提取元数据，防备有些旧数据没有这些字段
            emo = meta.get('emotion', '未知')
            impact = meta.get('impact_factor', 1)
            facts = meta.get('core_facts', [])
            
            # 记录历史对话片段，连同当时的元数据一起组装
            context_parts.append(
                f"---回忆片段---\n"
                f"[当时你的情绪: {emo} | 对你的冲击力: {impact}/10]\n"
                f"{d.page_content}"
            )
            
            # 收集事实
            if isinstance(facts, list):
                all_facts.extend(facts)
                
            # 找到这批记忆里冲击力最大的一条，它的情绪将主导你现在的回话语气
            if impact > max_impact:
                max_impact = impact
                dominant_emotion = emo
                
        # 组装动态变量
        context_str = "\n".join(context_parts) if context_parts else "（没有回忆起特别相关的往事）"
        
        # 事实去重并转为字符串
        unique_facts = list(set(all_facts))
        core_facts_str = "\n".join([f"- {f}" for f in unique_facts]) if unique_facts else "（无特异性事实）"
        
        # 3. 构造 Chain
        prompt = self.get_dynamic_prompt()
        chain = prompt | self.llm | StrOutputParser()
        
        history = self.chat_history[-20:]
        
        # 4. 执行流式输出，将所有动态变量喂入
        full_response = ""
        async for chunk in chain.astream({
            "persona_desc": self.persona_desc,
            "dominant_emotion": dominant_emotion,  # 🌟 注入主导情绪
            "core_facts": core_facts_str,          # 🌟 注入核心事实
            "context": context_str,
            "question": query,
            "chat_history": history
        }):
            full_response += chunk
            yield chunk, docs

        # 5. 保存近期记忆
        self.chat_history.append(HumanMessage(content=query))
        self.chat_history.append(AIMessage(content=full_response))

    def save_to_long_term_memory(self, user_query, ai_response):

        """
        后台静默将本轮对话存入向量库，更新性格分数，实现“全量实时进化”
        """
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        chat_name = os.getenv('CHAT_NAME', '梅子')
        combined_text = f"时间: {current_time}\n我: {user_query}\n{chat_name}: {ai_response}"
        
        # 封装为 LangChain Document 格式
        new_doc = Document(
            page_content=combined_text,
            metadata={"timestamp": current_time, "source": "live_chat"}
        )
        
        # 触发打分流水线 
        process_and_score_memories([new_doc]) 
        
        # 重新读取本地 JSON 画像，并实时刷新 Prompt 里的性格描述
        self.persona = load_global_persona()
        self.persona_desc = self._map_scores_to_text(self.persona)