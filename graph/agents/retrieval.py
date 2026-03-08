"""检索相关节点"""

from graph.state import AgentState
from rag.evaluator import AdaptiveRetriever, RetrievalGrade
from logger import logger
from config import config

from graph.agents.common import get_self_rag_evaluator


def adaptive_retrieve_node(state: AgentState) -> dict:
    """
    Self-RAG 自适应检索节点
    包含：初始检索 → 评估 → [改写/补充/通过]
    """
    try:
        from rag.retrieval_engine import RetrievalEngine

        engine = RetrievalEngine()
        adaptive = AdaptiveRetriever(engine, get_self_rag_evaluator())

        result = adaptive.retrieve_with_reflection(
            query=state["question"],
            iteration=0,
            previous_docs=state.get("retrieved_docs", []),
        )

        docs = result["documents"]
        if "grade" in result:
            grade = result["grade"]
        else:
            grade = (
                RetrievalGrade.PARTIALLY_RELEVANT if docs else RetrievalGrade.IRRELEVANT
            )

        if result["iterations"] > 0:
            observation = (
                f"Self-RAG: 经过{result['iterations']}轮反思检索，获得{len(docs)}篇文档"
            )
        else:
            observation = "Self-RAG: 首轮检索即获得高质量结果"

        logger.info(f"[Self-RAG] {observation}, 评级: {grade}")

        return {
            "retrieved_docs": docs,
            "retrieval_count": state.get("retrieval_count", 0)
            + result["iterations"]
            + 1,
            "next_step": "response_agent",
            "observation": observation,
            "retrieval_grade": grade,
        }

    except Exception as e:
        logger.error(f"[Self-RAG] 检索失败: {e}")
        return {"retrieved_docs": [], "next_step": "response_agent"}


def retrieval_agent_node(state: AgentState) -> dict:
    """检索Agent：负责企业知识库自适应检索"""
    if not config.ENABLE_SELF_RAG:
        try:
            from rag.retrieval_engine import RetrievalEngine

            engine = RetrievalEngine()
            docs = engine.hybrid_search(state["question"], top_k=5)
            return {
                "retrieved_docs": docs,
                "retrieval_count": state.get("retrieval_count", 0) + 1,
                "observation": f"普通RAG: 单次检索返回{len(docs)}篇文档",
                "retrieval_grade": None,
                "next_step": "response_agent",
                "active_agent": "retrieval_agent",
                "agent_outputs": [
                    {
                        "agent": "retrieval_agent",
                        "retrieved_docs": len(docs),
                        "retrieval_grade": None,
                        "mode": "baseline_rag",
                    }
                ],
            }
        except Exception as e:
            logger.error(f"[RAG] 普通检索失败: {e}")
            return {
                "retrieved_docs": [],
                "retrieval_count": state.get("retrieval_count", 0) + 1,
                "observation": f"普通RAG检索失败: {e}",
                "retrieval_grade": None,
                "next_step": "response_agent",
                "active_agent": "retrieval_agent",
                "agent_outputs": [
                    {
                        "agent": "retrieval_agent",
                        "retrieved_docs": 0,
                        "retrieval_grade": None,
                        "mode": "baseline_rag",
                    }
                ],
            }

    result = adaptive_retrieve_node(state)
    return {
        **result,
        "active_agent": "retrieval_agent",
        "agent_outputs": [
            {
                "agent": "retrieval_agent",
                "retrieved_docs": len(result.get("retrieved_docs", [])),
                "retrieval_grade": result.get("retrieval_grade"),
            }
        ],
    }
