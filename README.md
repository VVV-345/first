<<<<<<< HEAD
# 🎭 数字人格 — AI 陪伴系统

一个随时间进化的 AI 对话陪伴系统。每一次对话都会被 LLM 分析性格特征、存入向量记忆库，再用大五人格 EMA 算法增量更新 AI 的"灵魂画像"。下次聊天时，系统会自动检索相关回忆、带入最新人格状态，让 AI 越来越"像人"。

## 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                          main.py (总控台)                        │
│  ① 启动数据灌注流水线  ② 启动 Gradio Web UI                     │
└──────────────┬──────────────────────────────┬───────────────────┘
               │                              │
               ▼                              ▼
┌──────────────────────────┐    ┌─────────────────────────────────┐
│    ingestion/ 离线灌注    │    │          ui/ 在线对话            │
│  pipeline.py → tools.py  │    │     gradio_app.py (Gradio)       │
│  loader.py (聊天记录切分) │    │  左侧：人格/技能/角色设定        │
│  调用 LLM 批量评分 → 入库 │    │  右侧：流式聊天 + 检索回忆折叠   │
└──────────┬───────────────┘    └──────────────┬──────────────────┘
           │                                    │
           ▼                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      chat/engine.py (会话引擎)                    │
│  每次对话：检索 → 组装 prompt → 流式 LLM → 可选后台存长期记忆      │
└──────┬──────────────┬──────────────┬────────────────────────────┘
       │              │              │
       ▼              ▼              ▼
┌──────────┐ ┌──────────────┐ ┌──────────────────────────────────┐
│ retrieval│ │ chat/prompts │ │          scoring/scorer.py        │
│ hybrid.py│ │ 动态系统提示词 │ │  LLM 结构化评分 → MemoryFeatures  │
│ MMR+BM25 │ │ + 技能注入    │ │  大五人格 + 核心事实 + 冲击力     │
│ RRF 融合 │ └──────────────┘ └──────────────┬───────────────────┘
└────┬─────┘                                  │
     │                                        ▼
     ▼                               ┌────────────────────────────┐
┌────────────────────┐               │     persona/profile.py      │
│ qdrant_store.py    │               │  EMA 增量更新大五人格        │
│ HNSW 本地向量库     │               │  原子写入 personality.json   │
└────────────────────┘               └────────────────────────────┘
```

## 数据流向

```
用户发消息
  → engine.stream_chat()
    → hybrid retriever (MMR 向量 + BM25 关键词 → RRF 融合)
    → 拼接系统 prompt (人格描述 + 检索记忆 + 技能注入)
    → LLM 流式生成回复
    → (可选) save_to_long_term_memory()
      → scorer 评分 (MemoryFeatures)
      → EMA 更新人格
      → qdrant_store 入库 + BM25 热重构
```

## 各模块说明

### `config.py` — 全局配置

项目唯一的环境变量入口，`load_dotenv()` 只在此处调用一次。读取 LLM API Key/URL、嵌入模型路径、数据目录等。同时提供 `setup_logging()` 统一日志格式。

### `persona/profile.py` — 人格画像持久化

- `load_global_persona()` — 从 JSON 加载大五人格（5 维）+ 关系阶段 + 称呼规则，文件不存在时返回默认值
- `save_global_persona()` — 原子写入（先写 `.tmp` 再 rename，防崩溃损坏）
- `update_ema()` — **指数移动平均算法**，用对话冲击力系数作为学习率（α），增量更新人格维度。一次深刻的对话不会彻底颠覆人格，但会慢慢渗透

### `scoring/scorer.py` — LLM 结构化评分引擎

- `MemoryFeatures` (Pydantic) — 定义评分输出结构：瞬时情绪、核心事实、冲击力系数、大五人格打分
- `process_and_score_memories()` — 批量并发调用 LLM，提取每条对话的心理特征。含断点续传（`processed_registry.json`），支持分批处理大文件。每批处理完自动存档人格画像

### `retrieval/qdrant_store.py` — Qdrant 本地向量库

- `MemoryQdrantManager` — 管理 HNSW 索引的本地向量库，自动检测并清理死锁文件
- 支持创建 collection、动态获取嵌入维度、upsert 文档
- 使用 HuggingFace 嵌入模型编码文本

### `retrieval/hybrid.py` — 混合检索器

- `SimpleHybridRetriever` — **MMR 向量检索 + BM25 关键词检索**，通过 **RRF（倒数秩融合）算法**合并两种结果
- `get_hybrid_retriever()` — 工厂方法，扫描 Qdrant 全部记录构建 BM25 内存索引
- `add_documents()` — 增量添加新记忆并热重构 BM25 索引，无需重启

### `ingestion/loader.py` — 聊天记录加载器

- `TimeAwareChatLoader` — 按**时间间隔**（默认 5 分钟）或**对话轮次**（默认 10 句）自动切分聊天记录
- 支持时间戳头解析、系统消息过滤、重叠上下文

### `ingestion/pipeline.py` — Agent 调度灌注流水线

- `AgentIngestionPipeline` — 用本地 LLM（Ollama / Qwen）作为调度大脑，通过 `langchain.agents` 自主调用工具
- `run()` 方法接收自然语言指令，唤醒 Agent 执行数据清洗入库

### `ingestion/tools.py` — 灌注工具

- `execute_memory_ingestion_pipeline` — 供 Agent 调用的主工具函数，串联加载→评分→入库全流程

### `chat/prompts.py` — 动态 Prompt 模板

- `map_scores_to_text()` — 将大五人格数值映射为三档（高/中/低）自然语言描述
- `get_dynamic_prompt(active_skills)` — 构建完整系统提示词，包含：人格底色、表达习惯、关系阶段、瞬时情绪、核心事实、检索回忆、**技能注入**、禁止行为。返回 `(ChatPromptTemplate, skill_injections)`

### `chat/engine.py` — 对话引擎

- `PersonaEngine` — 核心类，管理 LLM、检索器、聊天历史、激活技能
- `stream_chat(query)` — **异步流式**主流程：检索 → 组装 prompt → LLM astream → yield 逐 token
- `save_to_long_term_memory()` — 后台静默：评分新对话 → 更新向量库 → 热刷新 BM25 → 重载人格
- `set_skills()` / `clear_skills()` — 技能插件管理，每次对话时注入到系统 prompt

### `skills/loader.py` — 技能加载器

- 定义 `.skill.json` 格式规范（`name` + `prompt_injection` 为必填项）
- `validate_skill()` / `load_skill()` — 校验并加载技能文件

### `skills/importer.py` — 技能导入器

- `detect_and_import()` — 自动识别三种格式：
  - `.skill.json` → 直接校验加载
  - Character Card v2 `.json`（SillyTavern 角色卡）→ 提取角色设定
  - `.txt` / `.md` → 全文作为 prompt_injection

### `ui/gradio_app.py` — Gradio Web 界面

- **左侧栏**：人格五维渐变进度条、角色设定（关系/称呼）、技能插件管理（拖入上传/移除/清除）
- **右侧对话区**：流式聊天（Thread+Queue 桥接异步→同步）、检索回忆以 HTML `<details>` 折叠嵌入 AI 消息
- **技能系统**：支持 `.skill.json` / Character Card v2 / 纯文本三种格式导入

## 核心设计特点

| 特性 | 实现 |
|------|------|
| 人格进化 | EMA 指数移动平均 + 冲击力加权学习率 |
| 记忆检索 | MMR 向量 + BM25 关键词 → RRF 融合 |
| 记忆持久化 | Qdrant 本地 HNSW 索引 + 增量 BM25 热重构 |
| 流式对话 | 独立线程 asyncio 事件循环 + Queue 桥接 |
| 技能插件 | 标准化 `.skill.json` + 兼容 Character Card v2 |
| 数据安全 | 原子写入（tmp + rename）、断点续传、锁文件清理 |

## 运行

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量（创建 .env 文件）
LLM_API_KEY=sk-your-key
BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL_PATH=BAAI/bge-small-zh-v1.5

# 完整启动（先灌注数据，再开 UI）
python main.py

# 或直接启动 UI
python ui/gradio_app.py
# 或通过 app.py（HuggingFace Spaces 兼容）
python app.py
```

## 技术栈

`Python` · `LangChain` · `Qdrant` · `Gradio` · `OpenAI API` · `Jieba` · `Sentence-Transformers` · `Pydantic`
=======
---
title: Digital Persona
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.16.0
python_version: '3.13'
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> 3a2b82f537c9c7a5e57f0622d6248f120654ef12
