# 项目根目录：main.py
import os
import sys
import subprocess
from langchain_core.callbacks import FileCallbackHandler
from core.ingestion_pipeline_agent import AgentIngestionPipeline

def main():
    print("========================================")
    print("🌸 数字人格系统核心总控台")
    print("========================================")

    # 1. 后台数据灌注(agent 版本)
    file_handler = FileCallbackHandler("ingestion_trace.log")
    pipeline = AgentIngestionPipeline(data_directory="input/data/",callbacks=[file_handler])

    pipeline.run("帮我看看 input/data/ 目录下有没有新数据需要清洗并洗入数据库。")
    
    

    # 2. 启动前端 UI
    ui_path = os.path.join("output", "web_ui.py")
    
    try:
        # 🌟 用 Popen获取子进程的控制权
        print("\n🌐 正在启动网页 UI，请稍候...")
        process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", ui_path])
        process.wait() # 挂起主程序，让网页一直跑
        
    except KeyboardInterrupt:
        # 🌟 捕捉到 Ctrl+C 强退
        print("\n🛑 接收到退出信号，正在安全释放数据库锁...")
        process.terminate() 
        process.wait()
        print("✅ 系统已彻底安全关闭。")

if __name__ == "__main__":
    main()