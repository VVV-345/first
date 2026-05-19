import streamlit as st
import asyncio
from output.chat_engine import PersonaEngine

st.set_page_config(page_title="数字人格对话", layout="wide")

# 🌟 1. 新增：使用 Streamlit 官方缓存装饰器，彻底锁死数据库连接实例
@st.cache_resource
def init_engine():
    print("⏳ 正在初始化全局唯一的 PersonaEngine...")
    return PersonaEngine()

# 🌟 2. 调用上面的缓存函数，而不是直接实例化
if "engine" not in st.session_state:
    st.session_state.engine = init_engine()

# 从缓存中获取引擎
engine = st.session_state.engine

# 侧边栏：性格展示与图片上传
with st.sidebar:
    st.title("🎭 人格内核")
    p = engine.persona
    
    st.write("大五人格切片：")
    st.progress(p.get('avg_openness', 5)/10, text=f"开放性: {p.get('avg_openness', 5):.1f}")
    st.progress(p.get('avg_extraversion', 5)/10, text=f"外向性: {p.get('avg_extraversion', 5):.1f}")
    st.progress(p.get('avg_conscientiousness', 5)/10, text=f"尽责性: {p.get('avg_conscientiousness', 5):.1f}")
    st.progress(p.get('avg_agreeableness', 5)/10, text=f"宜人性: {p.get('avg_agreeableness', 5):.1f}")
    st.progress(p.get('avg_neuroticism', 5)/10, text=f"神经质: {p.get('avg_neuroticism', 5):.1f}")
    
    st.divider()
    avatar = st.file_uploader("自定义头像", type=['png', 'jpg'])
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
        info_area = st.empty() # 专门留一个位置给检索信息
        
        # 🌟 使用 asyncio.run 处理异步生成器
        async def run_chat():
            full_response = ""
            async for content, docs in engine.stream_chat(prompt):
                full_response += content
                response_placeholder.markdown(full_response + "▌")
                
                # 渲染检索到的记忆参考
                if docs:
                    with info_area.expander("查看本次检索到的记忆"):
                        for d in docs: 
                            st.write(f"- {d.page_content}")
            
            response_placeholder.markdown(full_response)
            return full_response
            
        # 运行异步任务并获取完整回复
        final_answer = asyncio.run(run_chat())
        
        # 存入 UI 级别的聊天记录
        st.session_state.messages.append({"role": "assistant", "content": final_answer})
        
        # 如果开启了保存，在此处调用入库逻辑
        if save_toggle:
            st.toast("正在将本轮对话提炼并存入本地知识库...", icon="💾")
            # 这里准备对接你的 scorer_tool
            # engine.save_to_long_term_memory(prompt, final_answer)