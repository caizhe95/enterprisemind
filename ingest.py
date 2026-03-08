#!/usr/bin/env python3
"""知识库数据导入脚本 - Self-RAG元数据优化版"""

import os
import sys
import re
import hashlib
from pathlib import Path
from typing import List, Dict
from collections import Counter

from langchain_core.documents import Document

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from rag.document_loader import BatchDocumentLoader
from rag.document_processor import AdvancedDocumentProcessor
from rag.retrieval_engine import RetrievalEngine
from llm_factory import get_embeddings
from config import check_environment
from logger import logger


def _token_count(text: str) -> int:
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    english_words = len(re.findall(r"[a-zA-Z]+", text))
    return chinese_chars + english_words


def log_chunk_quality_metrics(chunks: List) -> None:
    """打印分块质量指标，便于调参与回归。"""
    if not chunks:
        logger.warning("⚠️ 无分块结果，跳过质量评估")
        return

    token_lengths = []
    strategy_counter = Counter()
    source_type_counter = Counter()

    for chunk in chunks:
        if isinstance(chunk, Document):
            content = chunk.page_content
            metadata = chunk.metadata or {}
        else:
            content = chunk.get("content", chunk.get("page_content", ""))
            metadata = chunk.get("metadata", {}) or {}

        token_lengths.append(_token_count(content))
        strategy_counter[metadata.get("processing_strategy", "unknown")] += 1
        source_type_counter[metadata.get("source_type", "unknown")] += 1

    sorted_lengths = sorted(token_lengths)
    n = len(sorted_lengths)
    p50 = sorted_lengths[n // 2]
    p95 = sorted_lengths[min(int(n * 0.95), n - 1)]
    avg_tokens = sum(sorted_lengths) / n

    logger.info("📊 分块质量指标:")
    logger.info(f"   - chunks: {n}")
    logger.info(
        f"   - token_length: avg={avg_tokens:.1f}, p50={p50}, p95={p95}, max={sorted_lengths[-1]}"
    )
    logger.info(f"   - strategy_dist: {dict(strategy_counter)}")
    logger.info(f"   - source_type_dist: {dict(source_type_counter)}")

    if avg_tokens < 80:
        logger.warning("⚠️ 平均chunk偏短，可能导致上下文碎片化")
    elif avg_tokens > 800:
        logger.warning("⚠️ 平均chunk偏长，可能影响召回精度")


def ensure_self_rag_metadata(documents: List[Dict]) -> List[Dict]:
    """确保文档包含Self-RAG所需的所有元数据"""
    processed = []

    for i, doc in enumerate(documents):
        if isinstance(doc, Document):
            doc_dict = {"content": doc.page_content, "metadata": doc.metadata}
        else:
            doc_dict = doc

        # 深拷贝避免修改原始数据
        doc_dict = {
            "content": doc_dict.get("content", doc_dict.get("page_content", "")),
            "metadata": {**doc_dict.get("metadata", {})},
        }

        meta = doc_dict["metadata"]

        # 核心元数据：文件名（用于溯源）
        if "file_name" not in meta:
            meta["file_name"] = meta.get("source", f"document_{i}")

        # 核心元数据：文档ID（用于去重和关联）
        if "chunk_id" not in meta:
            normalized = re.sub(r"\s+", " ", doc_dict["content"].strip())[:1200]
            content_hash = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
            base_name = Path(meta["file_name"]).stem
            meta["chunk_id"] = f"{base_name}_{content_hash}"

        # 核心元数据：来源类型
        if "source_type" not in meta:
            ext = Path(meta["file_name"]).suffix.lower()
            type_map = {
                ".pdf": "pdf",
                ".docx": "word",
                ".doc": "word",
                ".md": "markdown",
                ".txt": "text",
                ".csv": "csv",
                ".json": "json",
                ".html": "html",
            }
            meta["source_type"] = type_map.get(ext, "unknown")

        # Self-RAG评估字段初始化
        meta.setdefault(
            "retrieval_grade", None
        )  # highly_relevant / partially_relevant / irrelevant
        meta.setdefault("eval_reason", None)  # 评估理由
        meta.setdefault("rerank_score", 0.0)  # 重排序分数

        # 时间戳（用于时效性判断）
        import time

        meta.setdefault("indexed_at", time.time())

        processed.append(doc_dict)

    return processed


def init_knowledge_base(data_dir: str = "./data"):
    """初始化知识库主流程 - Self-RAG优化"""
    logger.info("🚀 开始初始化知识库（Self-RAG模式）...")

    # 环境检查
    try:
        config = check_environment()
    except Exception as e:
        logger.error(f"环境检查失败: {e}")
        return

    # 检查数据目录
    if not os.path.exists(data_dir):
        logger.error(f"❌ 数据目录 {data_dir} 不存在，请先创建并放入文档")
        return

    # 1. 批量加载文档
    logger.info(f"📂 正在扫描目录: {data_dir}")
    loader = BatchDocumentLoader(data_dir)
    raw_documents = loader.load()

    if not raw_documents:
        logger.error("❌ 未找到任何支持的文档（支持PDF/Word/Markdown/CSV/TXT/JSON）")
        return

    logger.info(f"✅ 成功加载 {len(raw_documents)} 个文档")

    # 打印文档统计
    type_count = {}
    for doc in raw_documents:
        t = doc.metadata.get("source_type", "unknown")
        type_count[t] = type_count.get(t, 0) + 1
    for t, c in type_count.items():
        logger.info(f"   - {t}: {c} 个")

    # 2. 智能分割处理
    logger.info("✂️  正在进行文档分割（结构感知）...")
    embedding_model = get_embeddings()
    processor = AdvancedDocumentProcessor(
        embedding_model=embedding_model, chunk_size=512
    )

    chunks = processor.process(raw_documents, strategy="auto")
    log_chunk_quality_metrics(chunks)

    # 3. Self-RAG元数据增强
    logger.info("🏷️  正在增强Self-RAG元数据...")
    chunks = ensure_self_rag_metadata(chunks)

    # 验证元数据完整性
    missing_metadata = 0
    for chunk in chunks:
        meta = chunk["metadata"]
        required = ["file_name", "chunk_id", "source_type", "retrieval_grade"]
        for field in required:
            if field not in meta:
                missing_metadata += 1
                logger.warning(
                    f"缺少元数据字段: {field} in {meta.get('file_name', 'unknown')}"
                )

    if missing_metadata == 0:
        logger.info("✅ 所有文档元数据完整")
    else:
        logger.warning(f"⚠️  {missing_metadata} 个字段缺失")

    chunk_id_counter = Counter(
        chunk["metadata"].get("chunk_id", "") for chunk in chunks
    )
    duplicate_ids = [
        cid for cid, count in chunk_id_counter.items() if cid and count > 1
    ]
    if duplicate_ids:
        logger.warning(f"⚠️ 检测到重复chunk_id: {len(duplicate_ids)} 个")
    else:
        logger.info("✅ chunk_id 唯一性检查通过")

    # 统计分割策略
    strategies = {}
    for chunk in chunks:
        s = chunk["metadata"].get("processing_strategy", "unknown")
        strategies[s] = strategies.get(s, 0) + 1

    logger.info(f"✅ 分割完成：共 {len(chunks)} 个片段")
    for s, count in strategies.items():
        logger.info(f"   - {s}: {count} 个")

    # 4. 存入向量库
    logger.info("💾 正在保存到向量数据库...")
    try:
        engine = RetrievalEngine(embedding_model=embedding_model)
        engine.add_documents(chunks)
        logger.info("✅ 向量数据库更新完成")
    except TypeError:
        logger.warning("⚠️ RetrievalEngine 不支持外部 embedding 模型，将使用内部初始化")
        engine = RetrievalEngine()
        engine.add_documents(chunks)
        logger.info("✅ 向量数据库更新完成")
    except Exception as e:
        logger.error(f"❌ 向量数据库保存失败: {e}")
        return

    # 5. Self-RAG验证测试
    logger.info("🔍 执行Self-RAG验证测试...")
    try:
        test_queries = ["智能手表Pro价格", "无线耳机功能", "销售额统计"]

        for query in test_queries:
            results = engine.hybrid_search(query, top_k=3)
            if results:
                top1 = results[0]
                meta = top1["metadata"]
                logger.info(f"✅ 查询 '{query}':")
                logger.info(f"   Top1: {meta.get('file_name')}")
                logger.info(f"   chunk_id: {meta.get('chunk_id')}")
                logger.info(f"   评估状态: {meta.get('retrieval_grade') or '待评估'}")

    except Exception as e:
        logger.warning(f"验证测试失败: {e}")

    logger.info("🎉 知识库初始化完成！Self-RAG已就绪")
    if config:
        logger.info(f"   存储路径: {config.CHROMA_PERSIST_DIR}")


def update_knowledge_base(data_dir: str = "./data"):
    """增量更新知识库（保留已有数据）"""
    logger.info("🔄 开始增量更新知识库...")

    if not os.path.exists(data_dir):
        logger.error(f"数据目录不存在: {data_dir}")
        return

    loader = BatchDocumentLoader(data_dir)
    documents = loader.load()

    if not documents:
        logger.warning("未找到新文档")
        return

    # 复用 embedding 模型
    embedding_model = get_embeddings()
    processor = AdvancedDocumentProcessor(embedding_model=embedding_model)
    chunks = processor.process(documents)
    log_chunk_quality_metrics(chunks)

    # Self-RAG元数据增强
    chunks = ensure_self_rag_metadata(chunks)

    try:
        engine = RetrievalEngine(embedding_model=embedding_model)
    except TypeError:
        engine = RetrievalEngine()

    engine.add_documents(chunks)

    logger.info(f"✅ 增量更新完成，新增 {len(chunks)} 个片段（已增强Self-RAG元数据）")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="企业知识库数据导入工具（Self-RAG版）")
    parser.add_argument("--data-dir", default="./data", help="数据目录路径")
    parser.add_argument("--update", action="store_true", help="增量更新模式")
    args = parser.parse_args()

    if args.update:
        update_knowledge_base(args.data_dir)
    else:
        init_knowledge_base(args.data_dir)
