"""HuggingFace Spaces 启动入口"""
import sys
import os

# 确保项目根目录在 sys.path
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from ui.gradio_app import create_ui

# HuggingFace Spaces 会自动检测 demo 变量并启动
demo = create_ui()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
