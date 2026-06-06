"""时间感知聊天记录加载器 — 按时间间隔/对话轮次自动切分"""
import os
import re
import logging
from datetime import datetime
from typing import Iterator
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader

from config import CHAT_NAME

logger = logging.getLogger(__name__)


class TimeAwareChatLoader(BaseLoader):
    """
    具备时间感知能力的聊天记录加载器。
    将指定目录下的本地文本文件，按5分钟时间间隔或10轮对话强制切分为独立的 Document 记忆块。
    """
    def __init__(self, directory_path: str, target_speaker=CHAT_NAME, max_gap_minutes=5, fallback_chunk_size=10, overlap_lines=2):
        self.directory_path = directory_path
        self.target_speaker = target_speaker
        self.max_gap_minutes = max_gap_minutes
        self.fallback_chunk_size = fallback_chunk_size
        self.overlap_lines = overlap_lines

        # 匹配标准时间戳头，例如 "2024-03-15 14:30:00 张三"
        self.header_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})\s+(.*)$')

    def lazy_load(self) -> Iterator[Document]:
        """
        使用 yield 实现懒加载生成器。
        有效防止读取超大历史记录文件时导致内存溢出。
        """
        if not os.path.exists(self.directory_path):
            logger.warning("数据目录不存在: %s", self.directory_path)
            return

        for filename in os.listdir(self.directory_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.directory_path, filename)
                logger.info("正在加载并切分文件: %s", filename)
                yield from self._process_single_file(file_path)

    def _process_single_file(self, file_path: str) -> Iterator[Document]:
        """核心切分逻辑"""
        current_chunk_lines = []
        last_time_obj = None
        block_start_time = None
        current_speaker = "未知"
        virtual_counter = 0

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # 过滤无意义的系统消息和空行
                    if not line or any(sys_msg in line for sys_msg in ["撤回", "拍了拍", "加入群聊", "通话"]):
                        continue

                    match = self.header_pattern.match(line)

                    # 匹配到带有时间戳的新消息
                    if match:
                        time_str = match.group(1)
                        current_speaker = match.group(2).strip()
                        try:
                            current_time_obj = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            continue

                        # 触发条件 1：时间间隔超过设定阈值（默认 5 分钟）
                        if last_time_obj and (current_time_obj - last_time_obj).total_seconds() > self.max_gap_minutes * 60:
                            if current_chunk_lines:
                                ts = block_start_time or f"Virtual_{virtual_counter:05d}"
                                yield Document(
                                    page_content="\n".join(current_chunk_lines),
                                    metadata={"timestamp": ts, "source": os.path.basename(file_path)}
                                )

                                current_chunk_lines = current_chunk_lines[-self.overlap_lines:] if self.overlap_lines > 0 else []
                                block_start_time = time_str
                                virtual_counter += 1

                        if not block_start_time:
                            block_start_time = time_str
                        last_time_obj = current_time_obj

                    # 匹配到无时间戳的纯人名行
                    elif line in [self.target_speaker, "我", "未知"]:
                        current_speaker = line

                    # 纯对话内容行
                    else:
                        current_chunk_lines.append(f"{current_speaker}: {line}")

                        # 触发条件 2：对话轮次达到兜底阈值（默认 10 句）
                        if len(current_chunk_lines) >= self.fallback_chunk_size:
                            ts = block_start_time or f"Virtual_{virtual_counter:05d}"
                            yield Document(
                                page_content="\n".join(current_chunk_lines),
                                metadata={"timestamp": ts, "source": os.path.basename(file_path)}
                            )
                            virtual_counter += 1

                            current_chunk_lines = current_chunk_lines[-self.overlap_lines:] if self.overlap_lines > 0 else []
                            block_start_time = None

            # 扫尾：如果文件读完还有剩余的对话，将其作为最后一块输出
            if current_chunk_lines:
                ts = block_start_time or f"Virtual_{virtual_counter:05d}"
                yield Document(
                    page_content="\n".join(current_chunk_lines),
                    metadata={"timestamp": ts, "source": os.path.basename(file_path)}
                )

        except Exception as e:
            logger.warning("读取或切分文件 %s 发生异常: %s", file_path, e)
