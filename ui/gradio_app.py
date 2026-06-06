"""Gradio Web UI — 数字人格对话界面 + 技能系统"""
import sys
import os

# 必须在所有项目内 import 之前：将项目根目录加入 sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import json
import queue
import threading
import asyncio
import gradio as gr

from chat.engine import PersonaEngine
from persona.profile import save_global_persona
from skills.importer import detect_and_import

# 获取当前文件所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 拼接 CSS 文件路径
CSS_PATH = os.path.join(CURRENT_DIR, "custom.css")
# 读取外部样式文件
with open(CSS_PATH, "r", encoding="utf-8") as f:
    CUSTOM_CSS = f.read()

# ============================================================
# 引擎（全局单例）
# ============================================================
engine = PersonaEngine()

# ============================================================
# 工具
# ============================================================
def _render_skills(skills: list[dict]) -> str:
    """渲染已激活技能为 Markdown"""
    if not skills:
        return "✦ 暂未激活技能"
    lines = ["✦ **已激活：**"]
    for s in skills:
        icon = s.get("icon", "🧩")
        name = s.get("name", "未命名")
        desc = s.get("description", "")[:60]
        line = f"- {icon} **{name}**"
        if desc:
            line += f" — _{desc}_"
        lines.append(line)
    return "\n".join(lines)

def _make_persona_html(p: dict) -> str:
    """生成人格展示 HTML"""
    traits = [
        ("开放性", "avg_openness", "fill-openness"),
        ("外向性", "avg_extraversion", "fill-extraversion"),
        ("尽责性", "avg_conscientiousness", "fill-conscientiousness"),
        ("宜人性", "avg_agreeableness", "fill-agreeableness"),
        ("神经质", "avg_neuroticism", "fill-neuroticism"),
    ]
    html = '<div class="sidebar-card"><h3>🎭 人格内核</h3>'
    for label, key, cls in traits:
        val = p.get(key, 5)
        pct = int(val * 10)
        html += f"""<div class="persona-bar">
            <span class="trait-label">{label}</span>
            <div class="trait-fill" style="background: rgba(255,255,255,0.06);">
                <div style="width:{pct}%;height:100%;border-radius:4px;" class="{cls}"></div>
            </div>
            <span class="trait-val">{val:.1f}</span>
        </div>"""
    html += "</div>"
    return html

def _build_memory_html(docs) -> str:
    """将检索文档转为可折叠 HTML，嵌入 AI 消息底部"""
    if not docs:
        return ""
    items = []
    for d in docs:
        text = d.page_content.strip()
        items.append(f'<div class="memory-item">💭 {text}</div>')
    return (
        '<details class="memory-details">'
        f'<summary>🔍 唤醒了 {len(docs)} 段相关回忆</summary>'
        + "\n".join(items) +
        '</details>'
    )

# ============================================================
# 技能管理事件
# ============================================================
def on_upload(file, active_skills):
    """拖入技能文件 → 自动识别格式并激活"""
    if file is None:
        return gr.skip(), active_skills, gr.skip()

    try:
        filepath = file if isinstance(file, str) else file.name
        with open(filepath, "rb") as f:
            content = f.read()
        filename = os.path.basename(filepath)
    except Exception:
        gr.Warning("⚠️ 无法读取文件")
        return gr.skip(), active_skills, gr.skip()

    skill = detect_and_import(content, filename)
    if skill is None:
        gr.Warning(f"⚠️ 无法识别技能格式: {filename}")
        return gr.skip(), active_skills, gr.skip()

    names = {s["name"] for s in active_skills}
    if skill["name"] in names:
        gr.Warning(f"⚠️ 技能 '{skill['name']}' 已激活")
        return gr.skip(), active_skills, gr.skip()

    new_skills = active_skills + [skill]
    md = _render_skills(new_skills)
    choices = [s["name"] for s in new_skills]
    gr.Info(f"✅ 技能 '{skill['name']}' 已激活！")
    return md, new_skills, gr.update(choices=choices)


def on_clear_skills(active_skills):
    """一键清除所有技能"""
    return "✦ 暂未激活技能", [], gr.update(choices=[])


def on_remove_skill(selected, active_skills):
    """移除指定技能"""
    if not selected or not active_skills:
        return gr.skip(), active_skills, gr.skip()
    new_skills = [s for s in active_skills if s["name"] != selected]
    md = _render_skills(new_skills)
    choices = [s["name"] for s in new_skills]
    return md, new_skills, gr.update(choices=choices)


# ============================================================
# 角色设定事件
# ============================================================
def on_update_relationship(new_val):
    engine.persona["relationship_status"] = new_val
    save_global_persona(engine.persona)


def on_update_nickname(new_val):
    engine.persona["nickname_rules"] = new_val
    save_global_persona(engine.persona)


# ============================================================
# 核心：流式对话（Thread + Queue 桥接异步 → Gradio 同步生成器）
# ============================================================
def respond(message, history, active_skills, save_toggle):
    """流式对话 — 检索回忆嵌入 AI 气泡内，可折叠展开"""
    if not message or not message.strip():
        yield history, "", active_skills
        return

    result_queue: queue.Queue = queue.Queue()

    async def _stream():
        engine.set_skills(active_skills)
        full = ""
        retrieved = None
        try:
            async for chunk, docs in engine.stream_chat(message):
                full += chunk
                if docs and retrieved is None:
                    retrieved = docs
                result_queue.put(("chunk", full, retrieved))
        except Exception:
            result_queue.put(("error", "（信号不太好，你刚才说什么来着？）", None))
        result_queue.put(("done", full, retrieved))

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_stream())
        finally:
            loop.close()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    # 初始化本轮 history
    history = list(history) if history else []
    history.append({"role": "user", "content": message})
    # 暂时放空内容，流式填充
    history.append({"role": "assistant", "content": ""})

    final_full = ""
    retrieved_docs = None

    while True:
        item = result_queue.get()
        status = item[0]

        if status == "error":
            final_full = item[1]
            history[-1]["content"] = final_full
            yield history, "", active_skills
            return

        if status == "done":
            final_full, retrieved_docs = item[1], item[2]
            break

        full, retrieved_docs = item[1], item[2]
        # 流式阶段：只更新纯文本，最后再加回忆 HTML
        history[-1]["content"] = full
        yield history, "", active_skills

    # 最后一帧：把检索回忆以可折叠 HTML 附加到 AI 回复末尾
    memory_html = _build_memory_html(retrieved_docs)
    history[-1]["content"] = final_full + memory_html
    yield history, "", active_skills

    # 后台保存长期记忆
    if save_toggle:
        try:
            engine.save_to_long_term_memory(message, final_full)
        except Exception:
            pass


# ============================================================
# 构建 UI
# ============================================================
def create_ui():
    p = engine.persona

    with gr.Blocks(title="数字人格对话", css=CUSTOM_CSS) as demo:
        gr.HTML(
            '<div class="main-header">'
            '<h1>🎭 数字人格</h1>'
            '<p>你的 AI 陪伴者，随时间进化的灵魂镜像</p>'
            '</div>'
        )

        with gr.Row(elem_classes=["main-row"]):
            # ═══════ 左侧栏（独立滚动）═══════
            with gr.Column(scale=1, min_width=300, elem_classes=["sidebar-col"]):
                # 人格内核（HTML 渐变进度条）
                persona_html = gr.HTML(_make_persona_html(p))

                # 角色设定
                with gr.Group():
                    gr.Markdown("### 👤 角色设定")
                    relationship = gr.Textbox(
                        label="💞 关系阶段",
                        value=p.get("relationship_status", "无话不谈的朋友"),
                        placeholder="如：正在冷战中 / 无话不谈",
                    )
                    nickname = gr.Textbox(
                        label="📛 对你的称呼",
                        value=p.get("nickname_rules", "不叫名字，直接说'你'"),
                        placeholder="如：叫你'喂' / 叫你'小家伙'",
                    )

                # 技能插件
                with gr.Group():
                    gr.Markdown("### 🧩 技能插件")
                    skill_file = gr.File(
                        label="📎 拖入技能文件",
                        file_types=[".json", ".txt", ".md"],
                    )
                    skills_display = gr.Markdown("✦ 暂未激活技能")
                    remove_skill_dd = gr.Dropdown(
                        label="选择要移除的技能",
                        choices=[],
                        interactive=True,
                    )
                    with gr.Row():
                        remove_skill_btn = gr.Button("🗑 移除选中", variant="secondary", size="sm")
                        clear_skills_btn = gr.Button("🧹 清除全部", variant="secondary", size="sm")

                save_toggle = gr.Checkbox(label="💾 自动保存到长期记忆", value=False)

            # ═══════ 右侧对话区 ═══════
            with gr.Column(scale=3, elem_classes=["chat-col"]):
                chatbot = gr.Chatbot(
                    height="100%",
                    label="",
                    placeholder="<div style='text-align:center;color:#666;padding:60px 0;'>"
                                "<div style='font-size:3rem;margin-bottom:12px;'>💬</div>"
                                "<p>发送一条消息，开启你们的对话……</p></div>",
                    elem_classes=["chatbot-container"],
                )
                with gr.Row():
                    msg = gr.Textbox(
                        label="",
                        placeholder="✏️  说点什么... 按 Enter 发送",
                        scale=9,
                        container=False,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)

        # ========== 状态 ==========
        active_skills_state = gr.State([])

        # ========== 事件绑定 ==========
        msg.submit(
            fn=respond,
            inputs=[msg, chatbot, active_skills_state, save_toggle],
            outputs=[chatbot, msg, active_skills_state],
        )
        send_btn.click(
            fn=respond,
            inputs=[msg, chatbot, active_skills_state, save_toggle],
            outputs=[chatbot, msg, active_skills_state],
        )

        skill_file.upload(
            fn=on_upload,
            inputs=[skill_file, active_skills_state],
            outputs=[skills_display, active_skills_state, remove_skill_dd],
        )

        clear_skills_btn.click(
            fn=on_clear_skills,
            inputs=[active_skills_state],
            outputs=[skills_display, active_skills_state, remove_skill_dd],
        )

        remove_skill_btn.click(
            fn=on_remove_skill,
            inputs=[remove_skill_dd, active_skills_state],
            outputs=[skills_display, active_skills_state, remove_skill_dd],
        )

        relationship.change(
            fn=on_update_relationship, inputs=[relationship], outputs=[],
        )
        nickname.change(
            fn=on_update_nickname, inputs=[nickname], outputs=[],
        )

    return demo


# ============================================================
# 启动入口
# ============================================================
if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
