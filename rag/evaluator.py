# self_rag/evaluator.py
"""Self-RAG 评估器 - 轻量质量控制闭环。"""

import re
from typing import List, Dict
from enum import Enum
from llm_factory import get_llm
from prompts.registry import PromptRegistry
from logger import logger


class RetrievalGrade(str, Enum):
    """检索相关性评级"""

    HIGHLY_RELEVANT = "highly_relevant"  # 直接包含答案
    PARTIALLY_RELEVANT = "partially_relevant"  # 部分相关，需补充
    IRRELEVANT = "irrelevant"  # 无关，需改写查询


class SupportGrade(str, Enum):
    """生成支持度评级"""

    FULLY_SUPPORTED = "fully_supported"  # 完全由检索内容支撑
    PARTIALLY_SUPPORTED = "partially_supported"  # 部分支撑，有推测
    NO_SUPPORT = "no_support"  # 无支撑，幻觉风险高


class SelfRAGEvaluator:
    """Self-RAG 自反思评估器"""

    def __init__(self):
        self.llm = get_llm()

    def evaluate_retrieval(
        self, question: str, documents: List[Dict], k: int = 3
    ) -> Dict:
        """
        评估检索质量，返回每个文档的相关性评级
        Self-RAG: 为每篇文档打标签 [relevant|irrelevant]
        """
        if not documents:
            return {"overall": RetrievalGrade.IRRELEVANT, "details": []}

        evaluated_docs = []
        has_relevant = False

        # 评估 Top-K 文档
        for i, doc in enumerate(documents[:k]):
            prompt = PromptRegistry.get(
                "self_rag_retrieval_eval",
                variables={
                    "question": question,
                    "document": doc["content"][:800],
                    "criteria": """
                评估标准：
                - HIGHLY_RELEVANT: 文档直接包含问题答案或关键数据
                - PARTIALLY_RELEVANT: 文档相关但不完整，需结合其他信息
                - IRRELEVANT: 文档与问题无关或完全无关
                """,
                },
            )

            try:
                result = self.llm.invoke(prompt).content.strip().lower()
                grade = self._parse_retrieval_grade(result)
                if grade in (
                    RetrievalGrade.HIGHLY_RELEVANT,
                    RetrievalGrade.PARTIALLY_RELEVANT,
                ):
                    has_relevant = True

                doc["metadata"]["retrieval_grade"] = grade
                doc["metadata"]["eval_reason"] = result
                evaluated_docs.append(doc)

            except Exception as e:
                logger.error(f"[Self-RAG] 检索评估失败: {e}")
                doc["metadata"]["retrieval_grade"] = RetrievalGrade.PARTIALLY_RELEVANT
                evaluated_docs.append(doc)

        # 整体决策逻辑
        highly_relevant_count = sum(
            1
            for d in evaluated_docs
            if d["metadata"].get("retrieval_grade") == RetrievalGrade.HIGHLY_RELEVANT
        )

        if highly_relevant_count >= 1:
            overall = RetrievalGrade.HIGHLY_RELEVANT
        elif has_relevant:
            overall = RetrievalGrade.PARTIALLY_RELEVANT
        else:
            overall = RetrievalGrade.IRRELEVANT

        retry_strategy = None
        if overall == RetrievalGrade.PARTIALLY_RELEVANT:
            retry_strategy = "supplement"
        elif overall == RetrievalGrade.IRRELEVANT:
            retry_strategy = "rewrite"

        return {
            "overall": overall,
            "details": evaluated_docs,
            "needs_retry": retry_strategy is not None,
            "retry_strategy": retry_strategy,
        }

    @staticmethod
    def _parse_retrieval_grade(result: str) -> RetrievalGrade:
        """稳健解析检索评级，避免 'irrelevant' 被 'relevant' 子串误判。"""
        text = (result or "").strip().lower()

        # 优先判定“无关”
        irrelevant_markers = ["irrelevant", "无关", "不相关", "完全无关"]
        if any(marker in text for marker in irrelevant_markers):
            return RetrievalGrade.IRRELEVANT

        # 再判定“部分相关”
        partial_markers = [
            "partially_relevant",
            "partially relevant",
            "partially",
            "部分相关",
            "部分",
        ]
        if any(marker in text for marker in partial_markers):
            return RetrievalGrade.PARTIALLY_RELEVANT

        # 最后判定“高度相关”
        high_markers = [
            "highly_relevant",
            "highly relevant",
            "highly",
            "高度相关",
            "直接相关",
            "relevant",
        ]
        if any(marker in text for marker in high_markers):
            return RetrievalGrade.HIGHLY_RELEVANT

        return RetrievalGrade.IRRELEVANT

    def evaluate_generation(
        self, question: str, answer: str, documents: List[Dict]
    ) -> Dict:
        """评估答案是否被证据支撑，用于幻觉风险控制。"""
        if not documents:
            return {
                "support_grade": SupportGrade.NO_SUPPORT,
                "support_reason": "no documents retrieved",
                "needs_regenerate": True,
                "is_hallucination_risk": True,
            }

        evidence = "\n\n".join(
            [
                f"[{doc['metadata'].get('file_name', 'doc')}]: {doc['content'][:500]}"
                for doc in documents[:3]
            ]
        )

        support_prompt = PromptRegistry.get(
            "self_rag_support_eval",
            variables={"question": question, "answer": answer, "evidence": evidence},
        )

        support_result = self.llm.invoke(support_prompt).content.strip().lower()

        if "full" in support_result or "完全" in support_result:
            support_grade = SupportGrade.FULLY_SUPPORTED
        elif "partial" in support_result or "部分" in support_result:
            support_grade = SupportGrade.PARTIALLY_SUPPORTED
        else:
            support_grade = SupportGrade.NO_SUPPORT

        return {
            "support_grade": support_grade,
            "support_reason": support_result,
            "needs_regenerate": support_grade == SupportGrade.NO_SUPPORT,
            "is_hallucination_risk": support_grade == SupportGrade.NO_SUPPORT,
        }

    def generate_reflection_query(
        self, original_query: str, previous_docs: List[Dict], feedback: str
    ) -> str:
        """在检索无关时生成一个更具体的新查询。"""
        previous_keywords = []
        for doc in previous_docs[:3]:
            content = doc.get("content", "")[:40].strip()
            if content:
                previous_keywords.append(content)

        prompt = PromptRegistry.get(
            "self_rag_rewrite",
            variables={
                "original_query": original_query,
                "feedback": feedback,
                "previous_keywords": " | ".join(previous_keywords),
            },
        )

        result = self.llm.invoke(prompt).content.strip()
        result = self._sanitize_reflection_query(result)

        if result.lower() == original_query.lower():
            result = f"{original_query} 详细信息"

        return result

    @staticmethod
    def _sanitize_reflection_query(query: str) -> str:
        """清洗改写查询，避免输出带编号/多行列表导致召回失败。"""
        if not query:
            return ""
        cleaned = query.replace("\r", "\n")
        lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
        if lines:
            cleaned = lines[0]
        cleaned = re.sub(r"^\d+[.)、\s-]+", "", cleaned).strip()
        return cleaned[:200]


class AdaptiveRetriever:
    """自适应检索器 - 结合Self-RAG评估动态调整策略"""

    def __init__(self, base_engine, evaluator: SelfRAGEvaluator):
        self.engine = base_engine
        self.evaluator = evaluator
        self.max_iterations = 2  # 防止无限循环

    def retrieve_with_reflection(
        self, query: str, iteration: int = 0, previous_docs: List[Dict] = None
    ) -> Dict:
        """检索后评估，不够好就补检索或改写查询。"""
        if iteration >= self.max_iterations:
            logger.warning("[Self-RAG] 达到最大迭代次数，返回当前最佳结果")
            final_docs = previous_docs or []
            if not final_docs:
                grade = RetrievalGrade.IRRELEVANT
            else:
                grade = self.evaluator.evaluate_retrieval(query, final_docs).get(
                    "overall", RetrievalGrade.PARTIALLY_RELEVANT
                )
            return {
                "documents": final_docs,
                "grade": grade,
                "final": True,
                "iterations": iteration,
            }

        # 执行检索
        docs = self.engine.hybrid_search(query, top_k=5)

        if previous_docs:
            # 去重合并
            seen = {d["content"][:100] for d in previous_docs}
            new_docs = [d for d in docs if d["content"][:100] not in seen]
            all_docs = previous_docs + new_docs
        else:
            all_docs = docs

        # Self-RAG 评估
        eval_result = self.evaluator.evaluate_retrieval(query, all_docs)

        if eval_result["overall"] == RetrievalGrade.HIGHLY_RELEVANT:
            logger.info("[Self-RAG] 检索质量高，直接返回")
            return {
                "documents": eval_result["details"],
                "grade": RetrievalGrade.HIGHLY_RELEVANT,
                "final": True,
                "iterations": iteration,
            }

        elif eval_result["overall"] == RetrievalGrade.PARTIALLY_RELEVANT:
            logger.info("[Self-RAG] 部分相关，补充检索...")
            expanded_query = f"{query} 详细参数 规格"
            return self.retrieve_with_reflection(
                expanded_query, iteration + 1, eval_result["details"]
            )

        else:
            logger.info("[Self-RAG] 检索无关，改写查询...")
            new_query = self.evaluator.generate_reflection_query(
                query,
                eval_result["details"],
                "检索结果与问题无关，需使用同义词或具体化",
            )
            return self.retrieve_with_reflection(
                new_query, iteration + 1, eval_result["details"]
            )
