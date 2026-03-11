"""检索相关节点（Worker ReAct版）"""

from __future__ import annotations

import re
from typing import Dict, List

from graph.state import AgentState
from logger import logger
from config import config
from graph.agents.worker_contract import build_worker_output
from graph.agents.field_utils import extract_fields_by_text
from graph.agents.section_utils import infer_section_targets, section_match_score
from tools.rerank_tool import rerank_tool


ENTITY_STOPWORDS = {
    "价格",
    "售价",
    "多少钱",
    "多少",
    "品类",
    "参数",
    "配置",
    "发布时间",
    "上市时间",
    "起售价",
    "续航",
    "电池",
    "容量",
    "对比",
    "哪个更贵",
    "哪个更便宜",
}


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


def _extract_query_entity(query: str) -> str:
    text = re.sub(r"[？?。！!]", "", query).strip()
    for marker in [
        "售价",
        "价格",
        "起售价",
        "品类",
        "参数",
        "配置",
        "发布时间",
        "上市时间",
        "续航",
        "电池",
        "容量",
        "多少钱",
        "是多少",
        "是什么",
    ]:
        if marker in text:
            text = text.split(marker)[0].strip()
            break
    text = re.sub(r"(现在|当前|目前|如今|现阶段)$", "", text).strip()
    text = re.sub(r"(的|请问|帮我|查下|查一下)$", "", text).strip()
    if text in ENTITY_STOPWORDS or len(text) < 2:
        return ""
    return text


def _extract_generation_number(text: str) -> str | None:
    match = re.search(r"(\d+)\s*代", text)
    if match:
        return match.group(1)
    return None


def _prioritize_entity_precise_docs(state: AgentState, query: str, docs: List[Dict]) -> List[Dict]:
    if not docs:
        return docs

    entity = _extract_query_entity(query)
    question_type = state.get("question_type")
    if not entity or question_type not in {"single_fact", "field_list", "comparison"}:
        return docs

    query_generation = _extract_generation_number(entity)

    def sort_key(doc: Dict) -> tuple[float, float, float]:
        meta = doc.get("metadata", {})
        file_name = str(meta.get("file_name") or meta.get("source") or "").lower()
        content = str(doc.get("content", ""))
        base = (
            float(meta.get("tool_rerank_score", 0) or 0)
            + float(meta.get("rerank_score", 0) or 0)
            + float(meta.get("rrf_score", 0) or 0)
        )

        exact_entity_bonus = 5.0 if entity and entity in content else 0.0
        prefix_bonus = 2.5 if entity and content.startswith(f"## {entity}") else 0.0

        source_bonus = 0.0
        if "products.md" in file_name:
            source_bonus += 2.5
        elif "policies.md" in file_name:
            source_bonus += 1.0
        elif "sales.md" in file_name:
            source_bonus -= 1.0

        generation_penalty = 0.0
        if query_generation:
            content_generation = _extract_generation_number(content)
            if content_generation and content_generation != query_generation:
                generation_penalty = -4.0

        return (exact_entity_bonus + prefix_bonus + source_bonus + generation_penalty, base, float(meta.get("search_latency_ms", 0) or 0) * -1)

    ranked = sorted(docs, key=sort_key, reverse=True)
    return ranked[:8]


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


def _section_queries_for_question(query: str) -> List[str]:
    return infer_section_targets(query, extract_fields_by_text(query))


def _merge_docs(primary: List[Dict], extra: List[Dict], top_k: int = 8) -> List[Dict]:
    merged: List[Dict] = []
    seen = set()
    for doc in primary + extra:
        meta = doc.get("metadata", {}) or {}
        doc_id = str(meta.get("chunk_id") or hash(str(doc.get("content", ""))[:120]))
        if doc_id in seen:
            continue
        seen.add(doc_id)
        merged.append(doc)
    return merged[:top_k]


def _prioritize_section_docs(query: str, docs: List[Dict]) -> List[Dict]:
    if not docs:
        return docs
    section_targets = _section_queries_for_question(query)
    if not section_targets:
        return docs

    def sort_key(doc: Dict) -> tuple[float, float]:
        meta = doc.get("metadata", {})
        file_name = str(meta.get("file_name") or meta.get("source") or "").lower()
        content = str(doc.get("content", ""))
        section_bonus = section_match_score(section_targets, content, meta)
        if "policies.md" in file_name:
            section_bonus += 2.0
        if "products.md" in file_name:
            section_bonus += 1.5
        base = (
            float(meta.get("tool_rerank_score", 0) or 0)
            + float(meta.get("rerank_score", 0) or 0)
            + float(meta.get("rrf_score", 0) or 0)
        )
        return (section_bonus, base)

    return sorted(docs, key=sort_key, reverse=True)[:8]

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
            if state.get("question_type") == "recommendation":
                query = _rewrite_query_for_react(base_query)
                thought = "候选证据不足，补充导购导向重写查询再检索一次。"
            else:
                query = base_query
                thought = "候选证据不足，保持原问题做第二轮检索，避免导购化改写干扰单事实题。"

        docs = engine.hybrid_search(query, top_k=8)
        extra_docs: List[Dict] = []
        for section_query in _section_queries_for_question(base_query):
            extra_docs.extend(engine.hybrid_search(section_query, top_k=4))
        if extra_docs:
            docs = _merge_docs(docs, extra_docs, top_k=12)
        docs = _prioritize_recommendation_docs(base_query, docs)
        docs = _prioritize_entity_precise_docs(state, base_query, docs)
        docs = _prioritize_section_docs(base_query, docs)
        used_rerank_tool = False
        if len(docs) > 3:
            reranked = rerank_tool.invoke({"query": base_query, "docs": docs, "top_k": 8})
            docs = reranked.get("docs", docs)
            used_rerank_tool = True
        docs = _prioritize_entity_precise_docs(state, base_query, docs)
        docs = _prioritize_section_docs(base_query, docs)
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
