from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from tools.data_process.ingestion_tool import execute_memory_ingestion_pipeline

class AgentIngestionPipeline:
    """Agent 调度版记忆灌注流水线。通过自然语言命令唤醒本地大模型，自主调用加工工具。"""
    def __init__(self, data_directory: str = "input/data/", callbacks=None):
        self.data_directory = data_directory
        self.callbacks = callbacks
        
        # 1. 初始化本地调度大脑 
        self.llm = ChatOpenAI(
            base_url="http://localhost:11434/v1", 
            api_key="ollama", 
            model="qwen2.5:7b",
            temperature=0.1,
            callbacks=self.callbacks  # 传递回调
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
            callbacks=self.callbacks
        )

    def run(self, command: str = "检查 input/data/ 目录，将新的聊天记录洗入数据库。"):
        print(f"🤖 [Agent Pipeline] 收到指令: {command}")
        
        try:
            # 让 Agent 纯粹做自然语言调度
            response = self.agent_executor.invoke({"input": command})
            print(f"✅ [Agent Pipeline] 任务汇报: {response['output']}")
            return True
        
        except Exception as e:
            print(f"❌ [Agent Pipeline] 调度失败: {e}")
            return False
