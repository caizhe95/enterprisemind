"""LLM工厂 - 中文Embedding优化版"""

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
import os

from cache.langchain_cache import initialize_llm_cache


class LLMFactory:
    """根据配置自动选择LLM和Embedding"""

    @staticmethod
    def get_chat_model() -> BaseChatModel:
        """获取聊天模型"""
        from config import config

        initialize_llm_cache()

        if config.RUN_MODE == "cloud":
            print(f"[LLM] 使用Ollama: {config.OLLAMA_MODEL}")
            return ChatOllama(
                model=config.OLLAMA_MODEL,
                base_url=config.OLLAMA_BASE_URL,
                temperature=0.3,
                num_ctx=8192,
            )

        # 本地：DeepSeek API
        if not config.DEEPSEEK_API_KEY:
            raise ValueError("本地模式需要 DEEPSEEK_API_KEY")

        print(f"[LLM] 使用DeepSeek API: {config.DEEPSEEK_MODEL}")
        return ChatOpenAI(
            model=config.DEEPSEEK_MODEL,
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_API_BASE,
            temperature=0.3,
            max_tokens=4096,
        )

    @staticmethod
    def get_embedding_model() -> Embeddings:
        """获取Embedding模型 - 失败即报错，不做降级"""
        from config import config

        if config.RUN_MODE == "cloud":
            # 云端：Ollama BGE-M3
            print(f"[Embedding] 使用Ollama: {config.OLLAMA_EMBEDDING_MODEL}")
            return OllamaEmbeddings(
                model=config.OLLAMA_EMBEDDING_MODEL,
                base_url=config.OLLAMA_BASE_URL,
            )

        # 本地：优先使用本地缓存模型，避免网络抖动导致初始化失败
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        model_name = "BAAI/bge-small-zh-v1.5"
        snapshot_root = "./models/models--BAAI--bge-small-zh-v1.5/snapshots"
        if os.path.isdir(snapshot_root):
            snapshot_dirs = [
                os.path.join(snapshot_root, d)
                for d in os.listdir(snapshot_root)
                if os.path.isdir(os.path.join(snapshot_root, d))
            ]
            if snapshot_dirs:
                model_name = snapshot_dirs[0]
        cache_dir = "./models"
        os.makedirs(cache_dir, exist_ok=True)

        print(f"[Embedding] 使用中文模型: {model_name}")

        try:
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
                cache_folder=cache_dir,
                multi_process=False,  # Windows兼容
            )
        except Exception as e:
            raise RuntimeError(
                "[Embedding] 初始化失败: 无降级策略。"
                "请检查网络/HF镜像可用性，或预先下载模型后重试。"
            ) from e


def get_llm() -> BaseChatModel:
    """便捷函数：获取LLM"""
    return LLMFactory.get_chat_model()


def get_embeddings() -> Embeddings:
    """便捷函数：获取Embedding"""
    return LLMFactory.get_embedding_model()
