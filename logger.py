"""轻量日志配置：统一输出、基础脱敏、错误分流。"""

import logging
import os
import re
import sys

from loguru import logger as _logger

os.makedirs("logs", exist_ok=True)
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
    """给标准 logging 增加脱敏，覆盖第三方库日志。"""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = _redact_secrets(record.getMessage())
            record.args = ()
        except Exception:
            pass
        return True


def _setup_stdlib_logging_guard():
    for name in ("langsmith", "langsmith.client"):
        lib_logger = logging.getLogger(name)
        lib_logger.setLevel(logging.ERROR)
        lib_logger.propagate = True

    root = logging.getLogger()
    redact_filter = _StdlibRedactFilter()
    root.addFilter(redact_filter)
    for handler in root.handlers:
        handler.addFilter(redact_filter)


_setup_stdlib_logging_guard()

_logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)

_logger.add(
    "logs/app_{time:YYYY-MM-DD}.log",
    rotation="100 MB",
    retention="30 days",
    encoding="utf-8",
    level="INFO",
    serialize=True,
)

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


def log_self_rag_step(
    step_type: str, reflection_count: int, grade: str, message: str, query: str = ""
):
    """兼容旧调用的 Self-RAG 日志入口。"""
    logger.bind(
        self_rag=True, reflection_count=reflection_count, grade=grade, query=query
    ).info(f"[{step_type}] {message}")


def log_retrieval_eval(
    question: str, grade: str, reason: str, reflection_count: int = 0
):
    """兼容旧调用的检索评估日志入口。"""
    logger.bind(
        self_rag=True, reflection_count=reflection_count, retrieval_grade=grade
    ).info(f"[RetrievalEval] Q: {question[:50]}... | Grade: {grade} | {reason}")


def log_generation_eval(
    support_grade: str, utility_grade: str, has_hallucination: bool
):
    """兼容旧调用的生成评估日志入口。"""
    logger.bind(
        self_rag=True,
        support_grade=support_grade,
        utility_grade=utility_grade,
        hallucination_risk=has_hallucination,
    ).info(
        f"[GenEval] Support: {support_grade}, Utility: {utility_grade}, Hallucination: {has_hallucination}"
    )
