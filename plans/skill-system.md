# 数字人格技能系统

## 设计概览

```
skills/                       ← 项目根目录
  __init__.py
  loader.py                   ← 加载 & 校验 .skill.json
  importer.py                 ← 把社区格式转成 .skill.json
  poet.skill.json             ← 示例技能

chat/prompts.py               ← get_dynamic_prompt() 新增 active_skills 参数
chat/engine.py                ← PersonaEngine 新增 set_skills() / clear_skills()
ui/app.py                     ← 侧边栏拖入区 + 技能卡片 + ❌ 移除
```

**技能如何生效**：激活的技能将其 `prompt_injection` 注入到 system prompt，不改引擎核心链路，纯增量。

---

## 1. 技能文件原生格式 (`.skill.json`)

```json
{
  "name": "诗人",
  "version": "1.0",
  "description": "以现代诗风格回复，多用意象和韵律",
  "icon": "🎭",
  "prompt_injection": "【技能激活：诗人模式】你现在是一位现代诗人。回复须有诗意和韵律感，多用意象和比喻。句子可自然断行。禁止过于直白的表达。"
}
```

| 字段 | 用途 |
|------|------|
| `name` | UI 展示名称 |
| `icon` | 卡片图标 |
| `description` | 悬停提示 |
| `prompt_injection` | 注入 system prompt 的核心指令 |
| `version` | 版本号（可选） |

---

## 2. 社区格式兼容（importer.py）

UI 拖入时自动识别文件类型：

| 输入 | 识别方式 | 提取逻辑 |
|------|----------|----------|
| `.skill.json` | 扩展名 | 直接通过 `loader.py` 校验 |
| Character Card v2 `.json` | 检查 JSON 顶层是否有 `spec` + `data` 字段 | 提取 `data.description` / `data.personality` / `data.scenario` → 拼成 `prompt_injection` |
| 纯文本 `.txt` 或 `.md` | 扩展名 | 全文作为 `prompt_injection`，`name` 取文件名 |

**Character Card v2 示例**（GitHub 上大量 SillyTavern 角色卡）：
```json
{
  "spec": "chara_card_v2",
  "data": {
    "name": "傲娇猫娘",
    "description": "一只修炼千年的猫妖，外表16岁少女...",
    "personality": "傲娇、毒舌但内心善良...",
    "first_mes": "哼！才不是特意等你的...",
    "scenario": "你捡到了一只受伤的猫..."
  }
}
```
→ 导入后自动生成：
```json
{
  "name": "傲娇猫娘",
  "icon": "🐱",
  "description": "一只修炼千年的猫妖，外表16岁少女...",
  "prompt_injection": "【角色设定】你是一只修炼千年的猫妖，外表16岁少女...\n【性格】傲娇、毒舌但内心善良...\n【场景】你捡到了一只受伤的猫..."
}
```

---

## 3. 新增文件

| 文件 | 用途 |
|------|------|
| `skills/__init__.py` | 空 |
| `skills/loader.py` | `load_skill(path)` — 加载 `.skill.json` 并校验必填字段；`validate_skill(dict)` — 格式校验 |
| `skills/importer.py` | `detect_and_import(file_bytes, filename)` — 自动识别格式 → 转成标准 skill dict；`import_chara_card(dict)` / `import_plain_text(text, name)` |
| `skills/poet.skill.json` | 示例 |

---

## 4. 修改文件

| 文件 | 改动 |
|------|------|
| `chat/prompts.py` | `get_dynamic_prompt(active_skills=None)` — system template 中新增 `{skill_injections}` 占位符，放在 `【⛔ 绝对禁止的 AI 行为】` 之前 |
| `chat/engine.py` | `PersonaEngine.__init__` 新增 `self.active_skills = []`；新增 `set_skills(skills)` / `clear_skills()`；`stream_chat()` 传 `active_skills` 给 prompt |
| `ui/app.py` | 侧边栏新增技能插件区域：① `st.file_uploader(type=["json","txt","md"])` 拖入区 ② 已激活技能卡片（图标+名称+❌） ③ 上传后调 `importer.detect_and_import()` → `engine.set_skills()` |

---

## 5. UI 交互

```
┌─────────────────────────────┐
│  🎭 人格内核                 │
│  [大五人格进度条...]          │
│─────────────────────────────│
│  🧩 技能插件                 │
│  ┌─────────────────────┐    │
│  │  拖入技能文件到此处    │   │  ← st.file_uploader
│  │  .json .txt .md      │   │     自动识别格式
│  └─────────────────────┘    │
│                              │
│  已激活:                     │
│  🎭 诗人 (原生)      ❌     │
│  🐱 傲娇猫娘 (导入)   ❌     │
└─────────────────────────────┘
```

---

## 6. Prompt 注入位置

```
【当前激活的技能】
🎭 诗人：你现在是现代诗人，回复须有诗意...
🐱 傲娇猫娘：你是一只猫妖，傲娇但内心善良...

【⛔ 绝对禁止的 AI 行为（违反即判定角色崩塌）】
...
```

---

## 7. 验证方式

1. 将 `poet.skill.json` 拖入 UI → 侧边栏显示 🎭 诗人卡片
2. 发送"给我写首诗" → LLM 回复带诗意风格
3. 拖入一个 Character Card v2 JSON → 自动识别并转换
4. 同时激活多个技能 → 都生效
5. 点 ❌ → 该技能移除，不再影响回复
