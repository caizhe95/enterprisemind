# logger.py
"""结构化日志配置 - Self-RAG增强版（修正版）"""

import sys
import os
import json
import re
import logging
from datetime import datetime
from loguru import logger as _logger

# 确保日志目录存在
os.makedirs("logs", exist_ok=True)

# 移除默认处理器
_logger.remove()


_SECRET_PATTERNS = [
    re.compile(r"(API\s*Key\s*:\s*)([^\s,;]+)", re.IGNORECASE),
    re.compile(r"(LANGCHAIN_API_KEY\s*=\s*)([^\s,;]+)", re.IGNORECASE),
    re.compile(r"(LANGSMITH_API_KEY\s*=\s*)([^\s,;]+)", re.IGNORECASE),
    re.compile(r"\blsv2_[A-Za-z0-9_\-]+\b"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]+\b"),
]


def _redact_secrets(text: str) -> str:
    if not text:
        return text

    output = text
    for pattern in _SECRET_PATTERNS:
        if pattern.groups >= 2:
            output = pattern.sub(r"\1[REDACTED]", output)
        else:
            output = pattern.sub("[REDACTED]", output)
    return output


class _StdlibRedactFilter(logging.Filter):
    """给标准 logging 增加敏感信息脱敏，覆盖第三方库日志（含 langsmith）。"""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            redacted = _redact_secrets(msg)
            record.msg = redacted
            record.args = ()
        except Exception:
            pass
        return True


def _setup_stdlib_logging_guard():
    # 压低 langsmith 日志噪声，避免网络抖动时刷屏并携带敏感字段
    for name in ("langsmith", "langsmith.client"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.CRITICAL)
        lg.propagate = True

    # 给 root 与现有 handler 统一挂脱敏过滤器
    root = logging.getLogger()
    root.addFilter(_StdlibRedactFilter())
    for h in root.handlers:
        h.addFilter(_StdlibRedactFilter())


_setup_stdlib_logging_guard()


def self_rag_sink(message):
    """自定义Self-RAG日志处理器（修正版）"""
    record = message.record

    # 提取Self-RAG特定字段
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "level": record["level"].name,
        "message": _redact_secrets(record["message"]),
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
        # Self-RAG特定字段（从extra中提取）
        "reflection_count": record["extra"].get("reflection_count", 0),
        "retrieval_grade": record["extra"].get("retrieval_grade"),
        "support_grade": record["extra"].get("support_grade"),
        "query": record["extra"].get("query"),
    }

    # 写入JSONL文件（按日期轮转）
    date_str = datetime.now().strftime("%Y-%m-%d")
    filepath = f"logs/self_rag_{date_str}.jsonl"

    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


# 1. 控制台输出（普通日志）
_logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
    filter=lambda record: not record["extra"].get(
        "self_rag", False
    ),  # 排除Self-RAG专用日志
)

# 2. Self-RAG专用控制台输出（带颜色标识）
_logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <magenta>🔄R{extra[reflection_count]: <1}</magenta> | "
    "<level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
    filter=lambda record: record["extra"].get("self_rag", False),  # 仅Self-RAG日志
)

# 3. 普通应用日志（JSON格式）
_logger.add(
    "logs/app_{time:YYYY-MM-DD}.log",
    rotation="100 MB",
    retention="30 days",
    encoding="utf-8",
    level="INFO",
    serialize=True,  # 标准JSON序列化
)

# 4. Self-RAG结构化日志（使用自定义sink）
_logger.add(
    self_rag_sink,  # 使用函数引用而非lambda
    level="INFO",
    filter=lambda record: record["extra"].get("self_rag", False),
)

# 5. 错误日志（文本格式）
_logger.add(
    "logs/error_{time:YYYY-MM-DD}.log",
    rotation="50 MB",
    retention="30 days",
    encoding="utf-8",
    level="ERROR",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}\n{exception}",
)

logger = _logger.patch(
    lambda record: record.update(message=_redact_secrets(record["message"]))
)


# Self-RAG日志便捷函数
def log_self_rag_step(
    step_type: str, reflection_count: int, grade: str, message: str, query: str = ""
):
    """记录Self-RAG反思步骤"""
    logger.bind(
        self_rag=True, reflection_count=reflection_count, grade=grade, query=query
    ).info(f"[{step_type}] {message}")


def log_retrieval_eval(
    question: str, grade: str, reason: str, reflection_count: int = 0
):
    """记录检索评估"""
    logger.bind(
        self_rag=True, reflection_count=reflection_count, retrieval_grade=grade
    ).info(f"[RetrievalEval] Q: {question[:50]}... | Grade: {grade} | {reason}")


def log_generation_eval(
    support_grade: str, utility_grade: str, has_hallucination: bool
):
    """记录生成评估"""
    logger.bind(
        self_rag=True,
        support_grade=support_grade,
        utility_grade=utility_grade,
        hallucination_risk=has_hallucination,
    ).info(
        f"[GenEval] Support: {support_grade}, Utility: {utility_grade}, Hallucination: {has_hallucination}"
    )
