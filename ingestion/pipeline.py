"""Agent 调度版记忆灌注流水线 — 通过自然语言唤醒本地大模型，自主调用加工工具"""
import logging
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from ingestion.tools import execute_memory_ingestion_pipeline
from config import LOCAL_LLM_BASE_URL, LOCAL_LLM_MODEL

logger = logging.getLogger(__name__)


class AgentIngestionPipeline:
    """Agent 调度版记忆灌注流水线。通过自然语言命令唤醒本地大模型，自主调用加工工具。"""
    def __init__(self, data_directory: str = "data/", callbacks=None):
        self.data_directory = data_directory
        self.callbacks = callbacks

        # 1. 初始化本地调度大脑
        self.llm = ChatOpenAI(
            base_url=LOCAL_LLM_BASE_URL,
            api_key="ollama",
            model=LOCAL_LLM_MODEL,
            temperature=0.1,
            max_retries=2,
            callbacks=self.callbacks
        )

        # 2. 挂载总控工具
        self.tools = [execute_memory_ingestion_pipeline]

        # 3. 设置 Agent 设定
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是数字人格后台数据管理员。根据用户需求，自主调用工具处理本地聊天记录并更新向量库。"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        # 4. 组装 Agent 执行器
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            timeout=300,
            max_retries=2,
            callbacks=self.callbacks
        )

    def run(self, command: str = "检查 data/ 目录，将新的聊天记录洗入数据库。"):
        logger.info("收到指令: %s", command)

        try:
            response = self.agent_executor.invoke({"input": command})
            logger.info("任务汇报: %s", response.get('output', ''))
            return True

        except Exception:
            logger.error("Agent 调度失败", exc_info=True)
            return False
