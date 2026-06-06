"""数字人格系统核心总控台"""
import os
import sys
import logging
import subprocess
from langchain_core.callbacks import FileCallbackHandler
from config import setup_logging, DATA_DIR
from ingestion.pipeline import AgentIngestionPipeline

setup_logging()
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 40)
    logger.info("🌸 数字人格系统核心总控台 启动")
    logger.info("=" * 40)

    # 1. 后台数据灌注 (Agent 版本)
    # 仅在本地有 Ollama 且 data/ 目录含历史聊天文件时运行
    # HuggingFace Spaces 上无本地 LLM 也无历史文件，自动跳过
    import glob
    txt_files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    if txt_files:
        logger.info("检测到 %d 个待处理聊天记录，启动数据灌注流水线", len(txt_files))
        try:
            file_handler = FileCallbackHandler("ingestion_trace.log")
            pipeline = AgentIngestionPipeline(data_directory=DATA_DIR, callbacks=[file_handler])
            pipeline.run("帮我看看 data/ 目录下有没有新数据需要清洗并洗入数据库。")
        except Exception:
            logger.warning("灌注流水线失败（可能缺少本地 LLM/Ollama），跳过", exc_info=True)
    else:
        logger.info("数据目录无待处理文件，跳过离线灌注（在线对话仍会自动记忆入库）")

    # 2. 启动前端 UI (Gradio)
    project_root = os.path.dirname(os.path.abspath(__file__))
    ui_path = os.path.join(project_root, "ui", "gradio_app.py")

    try:
        logger.info("正在启动 Gradio 网页 UI...")
        process = subprocess.Popen([sys.executable, ui_path], cwd=project_root)
    except FileNotFoundError:
        logger.error("未安装 gradio，请先执行 pip install gradio")
        sys.exit(1)

    try:
        returncode = process.wait()
        if returncode != 0:
            logger.warning("UI 进程退出，返回码: %d", returncode)
    except KeyboardInterrupt:
        logger.info("接收到退出信号，正在安全关闭 UI...")
        process.terminate()
        process.wait()
        logger.info("系统已彻底安全关闭。")


if __name__ == "__main__":
    main()
