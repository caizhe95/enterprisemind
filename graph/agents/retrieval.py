"""检索相关节点（Worker ReAct版）"""

from __future__ import annotations

import re
from typing import Dict, List

from graph.state import AgentState
from logger import logger
from config import config
from graph.agents.worker_contract import build_worker_output
from tools.rerank_tool import rerank_tool


def _need_shopping_slot_confirm(state: AgentState) -> bool:
    question = state.get("question", "")
    profile = state.get("shopping_profile") or {}
    has_profile = bool(profile.get("user_preference"))
    if has_profile:
        return False

    # 仅在导购语义问题下触发一次中断
    shopping_signals = ["推荐", "导购", "怎么选", "哪个好", "预算", "对比", "适合"]
    mentions_shopping = any(k in question for k in shopping_signals)
    first_round = int(state.get("retrieval_count", 0) or 0) == 0
    return mentions_shopping and first_round


def _rewrite_query_for_react(query: str) -> str:
    query = re.sub(r"\s+", " ", query).strip()
    if "用户补充偏好" in query:
        return query
    return f"{query}，请优先返回适合该需求的商品对比、价格区间和购买建议。"


def _score_docs(docs: List[Dict]) -> float:
    if not docs:
        return 0.0
    score = 0.0
    for doc in docs[:5]:
        meta = doc.get("metadata", {})
        score += float(meta.get("rerank_score", 0) or 0)
        score += float(meta.get("rrf_score", 0) or 0)
        score += float(meta.get("tool_rerank_score", 0) or 0)
    return score


def _prioritize_recommendation_docs(query: str, docs: List[Dict]) -> List[Dict]:
    if not docs:
        return docs
    recommendation_signals = ["推荐", "买哪个", "怎么选", "适合", "预算"]
    if not any(signal in query for signal in recommendation_signals):
        return docs

    def sort_key(doc: Dict) -> tuple[float, float]:
        meta = doc.get("metadata", {})
        file_name = str(meta.get("file_name") or meta.get("source") or "").lower()
        content = doc.get("content", "")
        guide_bonus = 3.0 if "guides" in file_name or "shopping_guide" in file_name else 0.0
        product_bonus = 2.0 if "products.md" in file_name else 0.0
        reason_bonus = 1.0 if ("推荐商品" in content or "推荐理由" in content) else 0.0
        base = float(meta.get("rerank_score", 0) or 0) + float(meta.get("rrf_score", 0) or 0)
        return (guide_bonus + product_bonus + reason_bonus, base)

    ranked = sorted(docs, key=sort_key, reverse=True)
    return ranked[:8]

def _react_retrieve(state: AgentState) -> dict:
    from rag.retrieval_engine import RetrievalEngine

    base_query = state.get("worker_input") or state["question"]
    max_steps = max(1, int(config.REACT_MAX_STEPS))
    engine = RetrievalEngine()
    best_docs: List[Dict] = []
    best_score = -1.0
    traces = []
    tool_calls = []

    for step in range(1, max_steps + 1):
        if step == 1:
            query = base_query
            thought = "先按用户原始问题进行混合检索，获取候选商品证据。"
        else:
            query = _rewrite_query_for_react(base_query)
            thought = "候选证据不足，补充导购导向重写查询再检索一次。"

        docs = engine.hybrid_search(query, top_k=8)
        docs = _prioritize_recommendation_docs(base_query, docs)
        used_rerank_tool = False
        if len(docs) > 3:
            reranked = rerank_tool.invoke({"query": base_query, "docs": docs, "top_k": 8})
            docs = reranked.get("docs", docs)
            used_rerank_tool = True
        current_score = _score_docs(docs)
        if current_score > best_score:
            best_score = current_score
            best_docs = docs

        traces.append(
            {
                "worker": "retrieval_agent",
                "step": step,
                "thought": thought,
                "action": "hybrid_search",
                "action_input": query,
                "observation": f"召回{len(docs)}条，score={current_score:.4f}",
            }
        )
        tool_calls.append(
            {
                "tool": "hybrid_search",
                "query": query,
                "docs": len(docs),
                "step": step,
            }
        )
        if used_rerank_tool:
            tool_calls.append(
                {
                    "tool": "rerank_tool",
                    "query": base_query,
                    "docs": len(docs),
                    "step": step,
                }
            )

        # 有有效结果且分数稳定后提前结束
        if len(docs) >= 5 and current_score >= max(0.05, best_score * 0.95):
            break

    grade = "highly_relevant" if len(best_docs) >= 5 else "partially_relevant"
    if not best_docs:
        grade = "irrelevant"

    return {
        "retrieved_docs": best_docs,
        "retrieval_count": state.get("retrieval_count", 0) + 1,
        "retrieval_grade": grade,
        "observation": f"ReAct检索完成，最终返回{len(best_docs)}条候选",
        "worker_trace": traces,
        "tool_calls": tool_calls,
        "step_count": state.get("step_count", 0) + len(traces),
        "next_step": "response_agent",
    }


def retrieval_agent_node(state: AgentState) -> dict:
    """检索Agent：导购优先的 Worker-ReAct 检索。"""
    if _need_shopping_slot_confirm(state):
        return {
            "next_step": "hitl_worker_confirm",
            "active_agent": "retrieval_agent",
            "hitl_request": {
                "type": "shopping_slot_confirm",
                "question": state["question"],
                "prompt": "请补充预算和偏好（如：3000内，轻薄续航优先）",
                "placeholder": "例如：预算3000-4000，偏好轻薄+长续航",
            },
            "worker_trace": [
                {
                    "worker": "retrieval_agent",
                    "step": 1,
                    "thought": "导购关键信息不足，先向用户确认预算和偏好。",
                    "action": "interrupt",
                    "observation": "触发shopping_slot_confirm",
                }
            ],
            "agent_outputs": [
                {
                    "agent": "retrieval_agent",
                    "status": "interrupted_for_slot",
                }
            ],
        }

    try:
        result = _react_retrieve(state)
        next_step = "judge" if state.get("execution_plan") else result.get("next_step", "response_agent")
        signals = []
        if result.get("retrieved_docs"):
            signals.append("documents_found")
        normalized_output = build_worker_output(
            worker="retrieval_agent",
            status="success" if result.get("retrieved_docs") else "partial",
            summary=result.get("observation", "检索完成"),
            artifacts={
                "retrieved_docs": result.get("retrieved_docs", []),
                "retrieval_grade": result.get("retrieval_grade"),
            },
            signals=list(dict.fromkeys(signals)),
            confidence=0.85 if result.get("retrieved_docs") else 0.3,
        )
        return {
            **result,
            "next_step": next_step,
            "active_agent": "retrieval_agent",
            "last_worker_output": normalized_output,
            "agent_outputs": [
                {
                    "agent": "retrieval_agent",
                    "retrieved_docs": len(result.get("retrieved_docs", [])),
                    "retrieval_grade": result.get("retrieval_grade"),
                    "mode": "worker_react",
                }
            ],
        }
    except Exception as e:
        logger.error(f"[RetrievalWorker] 检索失败: {e}")
        return {
            "retrieved_docs": [],
            "retrieval_count": state.get("retrieval_count", 0) + 1,
            "retrieval_grade": "irrelevant",
            "observation": f"检索失败: {e}",
            "next_step": "judge" if state.get("execution_plan") else "response_agent",
            "active_agent": "retrieval_agent",
            "last_worker_output": build_worker_output(
                worker="retrieval_agent",
                status="failed",
                summary=f"检索失败: {e}",
                artifacts={"retrieved_docs": [], "retrieval_grade": "irrelevant"},
                confidence=0.0,
                errors=[str(e)],
            ),
            "agent_outputs": [
                {
                    "agent": "retrieval_agent",
                    "retrieved_docs": 0,
                    "retrieval_grade": "irrelevant",
                    "mode": "worker_react",
                }
            ],
        }
