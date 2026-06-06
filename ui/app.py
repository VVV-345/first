"""Streamlit Web UI — 数字人格对话界面"""
import streamlit as st
import asyncio
import nest_asyncio
from chat.engine import PersonaEngine
from persona.profile import save_global_persona

nest_asyncio.apply()

st.set_page_config(page_title="数字人格对话", layout="wide")


@st.cache_resource
def init_engine():
    print("⏳ 正在初始化全局唯一的 PersonaEngine...")
    return PersonaEngine()


if "engine" not in st.session_state:
    st.session_state.engine = init_engine()

engine = st.session_state.engine

# 侧边栏：性格展示与图片上传
with st.sidebar:
    st.title("🎭 人格内核")
    p = engine.persona

    st.write("大五人格切片：")
    st.progress(p.get('avg_openness', 5) / 10, text=f"开放性: {p.get('avg_openness', 5):.1f}")
    st.progress(p.get('avg_extraversion', 5) / 10, text=f"外向性: {p.get('avg_extraversion', 5):.1f}")
    st.progress(p.get('avg_conscientiousness', 5) / 10, text=f"尽责性: {p.get('avg_conscientiousness', 5):.1f}")
    st.progress(p.get('avg_agreeableness', 5) / 10, text=f"宜人性: {p.get('avg_agreeableness', 5):.1f}")
    st.progress(p.get('avg_neuroticism', 5) / 10, text=f"神经质: {p.get('avg_neuroticism', 5):.1f}")

    st.divider()
    st.subheader("🎭 角色设定")

    new_relationship = st.text_input(
        "关系阶段",
        value=engine.persona.get("relationship_status", "无话不谈的朋友"),
        placeholder="如：正在冷战中 / 已经分开偶尔联系 / 无话不谈",
        key="rel_status_input"
    )
    new_nickname = st.text_input(
        "对你的称呼方式",
        value=engine.persona.get("nickname_rules", "不叫名字，直接说'你'"),
        placeholder="如：叫你'喂' / 叫你'小家伙'",
        key="nickname_input"
    )
    if new_relationship != engine.persona.get("relationship_status"):
        engine.persona["relationship_status"] = new_relationship
        save_global_persona(engine.persona)
    if new_nickname != engine.persona.get("nickname_rules"):
        engine.persona["nickname_rules"] = new_nickname
        save_global_persona(engine.persona)

    st.file_uploader("自定义头像", type=['png', 'jpg'])
    save_toggle = st.toggle("保存本轮对话到长期记忆", value=False)

# 主界面：对话区
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("想对她说点什么？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        info_area = st.empty()

        async def run_chat():
            full_response = ""
            async for content, docs in engine.stream_chat(prompt):
                full_response += content
                response_placeholder.markdown(full_response + "▌")

                if docs:
                    with info_area.expander("查看本次检索到的记忆"):
                        for d in docs:
                            st.write(f"- {d.page_content}")

            response_placeholder.markdown(full_response)
            return full_response

        final_answer = asyncio.run(run_chat())

        st.session_state.messages.append({"role": "assistant", "content": final_answer})
        # 防止浏览器端内存无限增长，保留最近 400 条消息
        if len(st.session_state.messages) > 400:
            st.session_state.messages = st.session_state.messages[-400:]

        if save_toggle:
            st.toast("正在将本轮对话提炼并存入本地知识库...", icon="💾")
            try:
                engine.save_to_long_term_memory(prompt, final_answer)
            except Exception:
                st.toast("保存失败，请稍后重试", icon="⚠️")
