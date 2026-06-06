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
    try:
        file_handler = FileCallbackHandler("ingestion_trace.log")
        pipeline = AgentIngestionPipeline(data_directory=DATA_DIR, callbacks=[file_handler])
        pipeline.run("帮我看看 data/ 目录下有没有新数据需要清洗并洗入数据库。")
    except Exception:
        logger.error("灌注流水线失败，继续启动 UI", exc_info=True)

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
