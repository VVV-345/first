"""集中配置模块 — 全项目只在此处调用 load_dotenv() 和 setup_logging()"""
import os
import logging
import warnings
from dotenv import load_dotenv

load_dotenv()  # 全项目唯一一次调用

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*Accessing.*from.*Returning.*")

_logging_initialized = False


def setup_logging(level=logging.INFO):
    """全项目只调一次，配置根 logger：控制台 + 文件双输出"""
    global _logging_initialized
    if _logging_initialized:
        return
    _logging_initialized = True
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(name)s | %(message)s",
        datefmt="%m-%d %H:%M:%S"
    )
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # 控制台：INFO 及以上
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt)
    root.addHandler(console)

    # 文件：DEBUG 及以上（全量记录到 app.log）
    file_handler = logging.FileHandler("app.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

LLM_MODEL = os.getenv("PROCESS_MODEL", "gpt-4o")
LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")
LLM_BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")
CHAT_NAME = os.getenv("CHAT_NAME", "数字人格")
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "BAAI/bge-m3-small-v1.5")
LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen2.5:7b")
DATA_DIR = os.getenv("DATA_DIR", "data")
