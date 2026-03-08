# retrieval_engine.py
"""检索引擎 - 集成Self-RAG元数据支持"""

from typing import List, Dict, Tuple, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
import numpy as np
from rank_bm25 import BM25Okapi
import re
import time

from llm_factory import get_embeddings
from config import config
from rag.query_enhancer import get_query_enhancer
from cache.cache_manager import cache_manager
from logger import logger


class RetrievalEngine:
    """多阶段检索引擎 - Self-RAG增强版"""

    def __init__(self, persist_dir: str = None, embedding_model=None):
        # 支持外部传入 embedding 模型，避免重复加载
        self.embeddings = embedding_model or get_embeddings()
        self.persist_dir = persist_dir or config.CHROMA_PERSIST_DIR

        self.vector_db = Chroma(
            persist_directory=self.persist_dir, embedding_function=self.embeddings
        )

        self.bm25: Optional[BM25Okapi] = None
        self.documents_cache: List[Dict] = []
        self.reranker = None
        self.enhancer = (
            get_query_enhancer() if config.ENABLE_QUERY_OPTIMIZATION else None
        )

        # Self-RAG新增：检索评估缓存（避免重复评估相同查询）
        self.eval_cache = {}

        # 检索缓存（内存级，10分钟过期）
        self.retrieval_cache = {}
        self._load_existing_documents()

    def _normalize_document(self, doc) -> Dict:
        """统一文档结构，确保元数据完整。"""
        if isinstance(doc, Document):
            doc_dict = {"content": doc.page_content, "metadata": dict(doc.metadata)}
        else:
            doc_dict = {
                "content": doc.get("content", doc.get("page_content", "")),
                "metadata": dict(doc.get("metadata", {})),
            }

        if "file_name" not in doc_dict["metadata"]:
            doc_dict["metadata"]["file_name"] = doc_dict["metadata"].get(
                "source", "unknown"
            )
        if "chunk_id" not in doc_dict["metadata"]:
            doc_dict["metadata"]["chunk_id"] = f"doc_{hash(doc_dict['content'][:50])}"

        doc_dict["metadata"].setdefault("retrieval_grade", None)
        doc_dict["metadata"].setdefault("eval_reason", None)
        return doc_dict

    def _load_existing_documents(self):
        """从已持久化的向量库恢复文档缓存，保证重启后 BM25 可用。"""
        try:
            existing = self.vector_db.get(include=["documents", "metadatas"])
        except Exception as e:
            logger.warning(f"[Recall] 加载本地向量库失败: {e}")
            return

        documents = existing.get("documents") or []
        metadatas = existing.get("metadatas") or []
        if not documents:
            return

        self.documents_cache = [
            self._normalize_document({"content": content, "metadata": metadata or {}})
            for content, metadata in zip(documents, metadatas)
            if content
        ]
        self._init_bm25()
        logger.info(f"[Recall] 已从向量库恢复 {len(self.documents_cache)} 条文档缓存")

    def add_documents(self, documents: List):
        """添加文档"""
        normalized_docs = [self._normalize_document(doc) for doc in documents]
        if not normalized_docs:
            return

        texts = [doc["content"] for doc in normalized_docs]
        metadatas = [doc["metadata"] for doc in normalized_docs]
        ids = [str(doc["metadata"]["chunk_id"]) for doc in normalized_docs]

        try:
            self.vector_db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        except Exception as e:
            logger.error(f"[Recall] 写入向量库失败: {e}")
            raise

        existing_ids = {
            str(doc["metadata"].get("chunk_id")) for doc in self.documents_cache
        }
        for doc in normalized_docs:
            if str(doc["metadata"]["chunk_id"]) not in existing_ids:
                self.documents_cache.append(doc)

        self._init_bm25()

    def _init_bm25(self):
        """初始化BM25"""
        if self.documents_cache:
            tokenized = [self._tokenize(d["content"]) for d in self.documents_cache]
            self.bm25 = BM25Okapi(tokenized)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z]+|\d+", text.lower())

    def multi_recall(self, query: str, top_k: int = 15) -> Tuple[List[Dict], Dict]:
        """多路召回"""
        candidates: List[Dict] = []
        stats = {"vector": 0, "bm25": 0, "exact": 0}

        # 向量召回
        try:
            results = self.vector_db.similarity_search_with_score(query, k=top_k)
            for doc, score in results:
                candidates.append(
                    {
                        "content": doc.page_content,
                        "metadata": {
                            **doc.metadata,
                            "vector_score": float(score),
                            "retrieval_grade": None,  # Self-RAG: 待评估
                            "eval_reason": None,
                        },
                        "recall_source": "vector",
                    }
                )
            stats["vector"] = len(results)
        except Exception as e:
            logger.error(f"[Recall] 向量失败: {e}")

        # BM25召回
        if self.bm25 and self.documents_cache:
            try:
                scores = self.bm25.get_scores(self._tokenize(query))
                top_indices = np.argsort(scores)[-top_k:][::-1]
                for idx in top_indices:
                    if scores[idx] > 0:
                        idx = int(idx)
                        doc: Dict = self.documents_cache[idx]
                        doc_content = doc.get("content", "")
                        if not doc_content:
                            continue

                        # 去重检查
                        is_duplicate = any(
                            c.get("content", "")[:100] == doc_content[:100]
                            for c in candidates
                        )

                        if not is_duplicate:
                            candidates.append(
                                {
                                    "content": doc_content,
                                    "metadata": {
                                        **doc.get("metadata", {}),
                                        "bm25_score": float(scores[idx]),
                                        "retrieval_grade": None,
                                        "eval_reason": None,
                                    },
                                    "recall_source": "bm25",
                                }
                            )
                stats["bm25"] = len([i for i in top_indices if scores[i] > 0])
            except Exception as e:
                logger.error(f"[Recall] BM25失败: {e}")

        return candidates, stats

    def reciprocal_rank_fusion(
        self, results_lists: List[List[Dict]], k: int = 60, top_n: int = 10
    ) -> List[Dict]:
        """RRF融合"""
        scores: Dict[str, Dict] = {}

        for results in results_lists:
            for rank, doc in enumerate(results):
                doc_id = doc.get("metadata", {}).get("chunk_id")
                if not doc_id:
                    doc_id = hash(doc.get("content", "")[:50])
                doc_id_str = str(doc_id)

                if doc_id_str not in scores:
                    scores[doc_id_str] = {"doc": doc, "score": 0, "sources": []}
                scores[doc_id_str]["score"] += 1.0 / (k + rank)
                scores[doc_id_str]["sources"].append(
                    doc.get("recall_source", "unknown")
                )

        fused = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [
            {
                "content": item["doc"]["content"],
                "metadata": {
                    **item["doc"].get("metadata", {}),
                    "rrf_score": round(item["score"], 4),
                    "fusion_sources": list(set(item["sources"])),
                    "retrieval_grade": None,  # Self-RAG: 初始未评估
                    "eval_reason": None,
                },
            }
            for item in fused[:top_n]
        ]

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """Cross-Encoder重排序"""
        if not candidates or not config.ENABLE_RERANK:
            return candidates

        try:
            from sentence_transformers import CrossEncoder

            if self.reranker is None:
                self.reranker = CrossEncoder(
                    config.RERANK_MODEL, device="cpu", max_length=512
                )

            pairs = [(query, doc.get("content", "")[:800]) for doc in candidates]
            scores = self.reranker.predict(pairs, batch_size=4)

            for doc, score in zip(candidates, scores):
                if "metadata" not in doc:
                    doc["metadata"] = {}
                doc["metadata"]["rerank_score"] = round(float(score), 4)
                # 保留Self-RAG字段
                doc["metadata"].setdefault("retrieval_grade", None)
                doc["metadata"].setdefault("eval_reason", None)

            return sorted(
                candidates,
                key=lambda x: x.get("metadata", {}).get("rerank_score", 0),
                reverse=True,
            )[:top_k]

        except Exception as e:
            logger.warning(f"[Rerank] 失败，使用RRF: {e}")
            return sorted(
                candidates,
                key=lambda x: x.get("metadata", {}).get("rrf_score", 0),
                reverse=True,
            )[:top_k]

    @cache_manager.cached(cache_type="memory", ttl=600, key_prefix="search")
    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        主入口：智能查询增强 → 多路召回 → RRF → 重排序
        返回结果包含Self-RAG所需完整元数据
        """
        start = time.time()

        # 1. 智能查询增强
        if config.ENABLE_QUERY_OPTIMIZATION and self.enhancer is not None:
            enhanced_queries = self.enhancer.enhance(query)
            logger.info(f"[Search] 查询优化开启: 1→{len(enhanced_queries)} 个变体")
        else:
            enhanced_queries = [query]
            logger.info("[Search] 查询优化关闭: 使用原始查询")

        # 2. 多路召回
        all_results = []
        for q in set(enhanced_queries):
            results, _ = self.multi_recall(q, top_k=15)
            all_results.append(results)

        # 3. RRF融合
        fused = self.reciprocal_rank_fusion(all_results, k=config.RRF_K, top_n=10)

        # 4. 精排
        final = self.rerank(query, fused, top_k=top_k)

        # 5. 添加元数据（包含Self-RAG字段）
        elapsed = (time.time() - start) * 1000
        for doc in final:
            if "metadata" not in doc:
                doc["metadata"] = {}
            doc["metadata"].update(
                {
                    "search_latency_ms": round(elapsed, 2),
                    "query_variants": len(enhanced_queries),
                    "retrieval_grade": None,  # 等待Self-RAG评估
                    "eval_reason": None,
                    "fusion_sources": doc["metadata"].get(
                        "fusion_sources", ["unknown"]
                    ),
                }
            )

        return final

    def get_document_by_id(self, chunk_id: str) -> Optional[Dict]:
        """通过ID获取文档（用于Self-RAG溯源）"""
        for doc in self.documents_cache:
            if doc.get("metadata", {}).get("chunk_id") == chunk_id:
                return doc
        return None
