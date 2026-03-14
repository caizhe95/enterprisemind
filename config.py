# config.py - 精简版（优先从.env读取）
import os
from pathlib import Path
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


class Config:
    """
    双模式配置：local (DeepSeek API) 或 cloud (Ollama本地)
    所有关键配置优先从环境变量读取，.env文件已覆盖全部
    """

    # ========== 1. 核心模式切换 ==========
    RUN_MODE = os.getenv("RUN_MODE", "local").strip().lower()  # local / cloud

    # ========== 2. LLM配置（两种模式） ==========
    # Local模式：DeepSeek API
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"  # 固定值，无需env

    # Cloud模式：Ollama
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:14b")
    OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3")

    # ========== 3. 数据存储（.env已定义） ==========
    DATABASE_URL = os.getenv(
        "DATABASE_URL", "mysql://root:123456@localhost:3306/enterprisemind"
    )
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    PERSISTENT_CACHE_BACKEND = os.getenv(
        "PERSISTENT_CACHE_BACKEND", "sqlite"
    ).strip().lower()
    PERSISTENT_CACHE_DB_PATH = os.getenv(
        "PERSISTENT_CACHE_DB_PATH", "./cache/persistent_cache.db"
    )
    PERSISTENT_CACHE_TTL_SECONDS = int(
        os.getenv("PERSISTENT_CACHE_TTL_SECONDS", "3600")
    )
    PERSISTENT_CACHE_REDIS_PREFIX = os.getenv(
        "PERSISTENT_CACHE_REDIS_PREFIX", "persistent_cache:"
    )
    LLM_CACHE_BACKEND = os.getenv("LLM_CACHE_BACKEND", "none").strip().lower()
    LLM_CACHE_TTL_SECONDS = int(os.getenv("LLM_CACHE_TTL_SECONDS", "3600"))
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    LLM_CACHE_FAIL_OPEN = os.getenv("LLM_CACHE_FAIL_OPEN", "true").lower() == "true"

    # ========== 4. 可选功能 ==========
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

    # LangSmith（可选，保持现状）
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "enterprisemind-demo")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_ENDPOINT = os.getenv(
        "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
    )
    LANGSMITH_FAIL_OPEN = os.getenv("LANGSMITH_FAIL_OPEN", "true").lower() == "true"
    LANGSMITH_PRECHECK_TIMEOUT_SEC = float(
        os.getenv("LANGSMITH_PRECHECK_TIMEOUT_SEC", "2.0")
    )

    # ========== 5. 服务配置（.env已定义） ==========
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

    # ========== 6. HITL交互阈值 ==========
    HITL_AMBIGUITY_CONFIDENCE_LEVEL = os.getenv(
        "HITL_AMBIGUITY_CONFIDENCE_LEVEL", "medium"
    ).lower()
    HITL_REQUIRE_CONFIRM_ON_DUAL_ROUTE = (
        os.getenv("HITL_REQUIRE_CONFIRM_ON_DUAL_ROUTE", "true").lower() == "true"
    )
    HITL_ENABLE_LOW_CONF_CONFIRM = (
        os.getenv("HITL_ENABLE_LOW_CONF_CONFIRM", "true").lower() == "true"
    )

    # ========== 7. 本地Embedding备用（local模式使用） ==========
    # 这些保持硬编码，因为.env没定义，且是技术细节
    LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"  # 384维
    HF_ENDPOINT = "https://hf-mirror.com"  # 国内镜像加速

    # ========== 8. 功能开关（代码内固定） ==========
    ENABLE_RERANK = os.getenv("ENABLE_RERANK", "true").lower() == "true"  # 重排序开关
    ENABLE_SELF_RAG = (
        os.getenv("ENABLE_SELF_RAG", "true").lower() == "true"
    )  # Self-RAG总开关
    ENABLE_QUERY_OPTIMIZATION = (
        os.getenv("ENABLE_QUERY_OPTIMIZATION", "true").lower() == "true"
    )  # 查询优化开关
    ENABLE_LLM_INTENT_ROUTING = (
        os.getenv("ENABLE_LLM_INTENT_ROUTING", "true").lower() == "true"
    )
    # 默认使用更适合中文场景的 reranker，可通过 .env 覆盖
    RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
    RRF_K = 60  # RRF融合参数

    # ========== 9. 智能导购工作流 ==========
    AGENT_MODE = os.getenv("AGENT_MODE", "worker_react").lower()
    ENABLE_SELF_RAG_GUARD = (
        os.getenv("ENABLE_SELF_RAG_GUARD", "true").lower() == "true"
    )
    REACT_MAX_STEPS = int(os.getenv("REACT_MAX_STEPS", "3"))


config = Config()


def check_environment():
    """启动前环境检查（简化版）"""
    if config.RUN_MODE not in {"local", "cloud"}:
        raise ValueError(
            f"RUN_MODE 仅支持 'local' 或 'cloud'，当前值: {config.RUN_MODE}"
        )

    print(f"\n{'=' * 50}")
    print("🚀 智能导购多 Agent 系统启动配置")
    print(f"{'=' * 50}")
    print(f"模式: {config.RUN_MODE.upper()}")

    if config.RUN_MODE == "cloud":
        print(f"LLM: Ollama ({config.OLLAMA_MODEL})")
        print(f"Embedding: {config.OLLAMA_EMBEDDING_MODEL}")
        # 检查Ollama连接...
    else:
        print(f"LLM: DeepSeek API ({config.DEEPSEEK_MODEL})")
        print(f"Embedding: {config.LOCAL_EMBEDDING_MODEL}")
        if not config.DEEPSEEK_API_KEY:
            print("⚠️ 警告: DEEPSEEK_API_KEY 未设置")

    print(f"向量库: {config.CHROMA_PERSIST_DIR}")
    print(f"数据库: {config.DATABASE_URL.split('@')[-1]}")  # 隐藏密码
    print(f"业务缓存: {config.PERSISTENT_CACHE_BACKEND}")
    print(f"LLM缓存: {config.LLM_CACHE_BACKEND}")
    print(f"默认服务配置: http://{config.HOST}:{config.PORT}")
    print("FastAPI 实际监听地址以 Uvicorn 日志中的 'running on' 为准")
    print(f"调试模式: {'开启' if config.DEBUG else '关闭'}")
    print(f"Self-RAG: {'开启' if config.ENABLE_SELF_RAG else '关闭'}")
    print(f"Self-RAG Guard: {'开启' if config.ENABLE_SELF_RAG_GUARD else '关闭'}")
    print(f"查询优化: {'开启' if config.ENABLE_QUERY_OPTIMIZATION else '关闭'}")
    print(f"Agent模式: {config.AGENT_MODE}")
    print(
        "HITL: ambiguity<={level}, dual_route_confirm={dual}, low_conf_confirm={low}".format(
            level=config.HITL_AMBIGUITY_CONFIDENCE_LEVEL,
            dual=config.HITL_REQUIRE_CONFIRM_ON_DUAL_ROUTE,
            low=config.HITL_ENABLE_LOW_CONF_CONFIRM,
        )
    )
    print(f"{'=' * 50}\n")

    return config
