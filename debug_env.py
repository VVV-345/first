import sys
import os

print(f"当前 Python 解释器: {sys.executable}")
print("\n--- 模块搜索路径 ---")
for p in sys.path:
    print(f" - {p}")

try:
    import langchain
    print(f"\n✅ 找到了 LangChain，路径为: {langchain.__file__}")
    # 尝试访问 memory
    from langchain.memory import ConversationBufferWindowMemory
    print("✅ 成功导入 ConversationBufferWindowMemory")
except Exception as e:
    print(f"\n❌ 报错了: {e}")