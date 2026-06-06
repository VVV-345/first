"""数字人格聊天引擎 — 检索 + Prompt 组装 + 流式输出 + 长期记忆持久化"""
import time
import logging
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

from config import LLM_MODEL, LLM_API_KEY, LLM_BASE_URL, CHAT_NAME
from retrieval.hybrid import get_hybrid_retriever
from persona.profile import load_global_persona, DEFAULT_RELATIONSHIP, DEFAULT_NICKNAME
from scoring.scorer import process_and_score_memories
from chat.prompts import map_scores_to_text, get_dynamic_prompt

logger = logging.getLogger(__name__)


class PersonaEngine:
    def __init__(self):
        self.persona = load_global_persona()
        self.retriever = get_hybrid_retriever()
        self.persona_desc = map_scores_to_text(self.persona)

        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            streaming=True,
            temperature=0.7
        )
        self.chat_history = []

    def _parse_retrieved_docs(self, docs):
        """从检索到的文档中提取：上下文文本、核心事实、主导情绪、最大冲击力"""
        context_parts = []
        all_facts = []
        max_impact = 0
        dominant_emotion = "平静"

        for d in docs:
            meta = d.metadata
            emo = meta.get('instant_emotion', '未知')
            impact = meta.get('impact_factor', 0.01)
            facts = meta.get('core_facts', [])

            context_parts.append(
                f"---回忆片段---\n"
                f"[当时你的情绪: {emo} | 对你的冲击力: {impact * 20:.1f}/10]\n"
                f"{d.page_content}"
            )

            if isinstance(facts, list):
                all_facts.extend(facts)

            if impact > max_impact:
                max_impact = impact
                dominant_emotion = emo

        context_str = "\n".join(context_parts) if context_parts else "（没有回忆起特别相关的往事）"
        unique_facts = list(set(all_facts))
        core_facts_str = "\n".join([f"- {f}" for f in unique_facts]) if unique_facts else "（无特异性事实）"

        return context_str, core_facts_str, dominant_emotion

    async def stream_chat(self, query: str):
        # 1. 检索向量库（异步调用）
        docs = await self.retriever.ainvoke(query)

        # 2. 动态元数据解析
        context_str, core_facts_str, dominant_emotion = self._parse_retrieved_docs(docs)

        # 3. 构造 Chain
        prompt = get_dynamic_prompt()
        chain = prompt | self.llm | StrOutputParser()

        history = self.chat_history[-20:]

        # 4. 执行流式输出
        full_response = ""
        try:
            async for chunk in chain.astream({
                "persona_desc": self.persona_desc,
                "dominant_emotion": dominant_emotion,
                "core_facts": core_facts_str,
                "context": context_str,
                "question": query,
                "chat_history": history,
                "persona_name": CHAT_NAME,
                "nickname_rules": self.persona.get("nickname_rules", DEFAULT_NICKNAME),
                "current_relationship_status": self.persona.get("relationship_status", DEFAULT_RELATIONSHIP),
            }):
                full_response += chunk
                yield chunk, docs
        except Exception:
            logger.error("LLM 流式调用失败", exc_info=True)
            full_response = "（内心有点乱，稍等一下就好...）"
            yield full_response, docs

        # 5. 保存近期记忆
        self.chat_history.append(HumanMessage(content=query))
        self.chat_history.append(AIMessage(content=full_response))
        # 只保留最近 200 轮对话，防止内存无限增长
        if len(self.chat_history) > 200:
            self.chat_history = self.chat_history[-200:]

    def save_to_long_term_memory(self, user_query, ai_response):
        """
        后台静默将本轮对话存入向量库，更新性格分数，同步刷新内存检索器分支
        """
        try:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            combined_text = f"时间: {current_time}\n我: {user_query}\n{CHAT_NAME}: {ai_response}"

            new_doc = Document(
                page_content=combined_text,
                metadata={"timestamp": current_time, "source": "live_chat"}
            )

            # 1. 产生人格评分并打上特征元数据快照
            scored_docs = process_and_score_memories([new_doc])

            if scored_docs:
                # 2. 创建本地客户端，真正将新对话落盘入向量库
                from retrieval.qdrant_store import MemoryQdrantManager
                db = MemoryQdrantManager()
                db.upsert_memory_chunks(scored_docs)
                db.close()  # 关闭连接释放文件锁，防止进程级锁死

                # 3. 同步刷新前台服务常驻内存的 BM25 检索分支
                if hasattr(self.retriever, "add_documents"):
                    # 转换解构为 BM25 可接收的纯净文档（附带清洗完的高级人格元数据）
                    bm25_new_docs = [
                        Document(page_content=d.page_content, metadata=d.metadata)
                        for d in scored_docs
                    ]
                    self.retriever.add_documents(bm25_new_docs)

            # 4. 刷新内存中缓存的性格状态描述
            self.persona = load_global_persona()
            self.persona_desc = map_scores_to_text(self.persona)
            logger.info("🌸 系统性格和检索双通道已完成增量进化。")

        except Exception:
            logger.error("长期记忆保存失败", exc_info=True)
