"""LLM工厂 - 中文Embedding优化版"""

from pathlib import Path
import hashlib
import math
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
import os


class LLMFactory:
    """根据配置自动选择LLM和Embedding"""

    @staticmethod
    def get_chat_model() -> BaseChatModel:
        """获取聊天模型"""
        from config import config

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
        """获取Embedding模型 - BGE中文优化版"""
        from config import config

        if config.RUN_MODE == "cloud":
            # 云端：Ollama BGE-M3
            print(f"[Embedding] 使用Ollama: {config.OLLAMA_EMBEDDING_MODEL}")
            return OllamaEmbeddings(
                model=config.OLLAMA_EMBEDDING_MODEL,
                base_url=config.OLLAMA_BASE_URL,
            )

        # 本地：优先中文模型，失败时回退到本地已缓存模型
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        os.environ["HF_HUB_OFFLINE"] = "0"

        model_name = "BAAI/bge-small-zh-v1.5"
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
            print(f"[Embedding] 中文模型不可用，尝试本地回退: {e}")

        local_snapshot = (
            Path(cache_dir)
            / "models"
            / "models--sentence-transformers--all-MiniLM-L6-v2"
            / "snapshots"
        )
        if local_snapshot.exists():
            snapshots = sorted([p for p in local_snapshot.iterdir() if p.is_dir()])
            if snapshots:
                fallback_path = str(snapshots[0])
                print(f"[Embedding] 使用本地回退模型: {fallback_path}")
                return HuggingFaceEmbeddings(
                    model_name=fallback_path,
                    model_kwargs={"device": "cpu", "local_files_only": True},
                    encode_kwargs={"normalize_embeddings": True},
                    cache_folder=cache_dir,
                    multi_process=False,
                )

        print("[Embedding] 未找到可用的本地回退模型")
        print("[Embedding] 使用本地哈希回退向量")
        return HashEmbeddings()


class HashEmbeddings(Embeddings):
    """纯本地回退向量，保证无外部依赖时系统仍可运行。"""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = text.lower().split()
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dimension
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_llm() -> BaseChatModel:
    """便捷函数：获取LLM"""
    return LLMFactory.get_chat_model()


def get_embeddings() -> Embeddings:
    """便捷函数：获取Embedding"""
    return LLMFactory.get_embedding_model()
