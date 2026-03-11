"""Recommendation worker for shopping guidance questions."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from graph.state import AgentState
from graph.agents.worker_contract import build_worker_output
from tools.candidate_ranker import candidate_ranker
from tools.catalog_filter import catalog_filter


def _parse_budget(question: str) -> int | None:
    match = re.search(r"预算\s*(\d+)", question)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)\s*元", question)
    if match and "预算" in question:
        return int(match.group(1))
    return None


def _infer_category(question: str) -> str | None:
    categories = ["手机", "笔记本", "平板", "智能手表", "蓝牙耳机", "电视", "空调", "冰箱"]
    for category in categories:
        if category in question:
            return category
    return None


def _infer_preferences(question: str) -> List[str]:
    keywords = [
        "轻薄",
        "续航",
        "性能",
        "拍照",
        "影像",
        "游戏",
        "办公",
        "学生",
        "护眼",
        "性价比",
        "散热",
    ]
    return [kw for kw in keywords if kw in question]


def _infer_scenarios(question: str) -> List[str]:
    scenario_keywords = {
        "办公": ["办公", "通勤", "文档", "会议"],
        "学生": ["学生", "上课", "学习", "宿舍"],
        "游戏": ["游戏", "高性能", "电竞"],
        "拍照": ["拍照", "影像", "摄影"],
        "轻薄续航": ["轻薄", "续航", "便携", "出差"],
    }
    scenarios = []
    for scenario, keywords in scenario_keywords.items():
        if any(keyword in question for keyword in keywords):
            scenarios.append(scenario)
    return scenarios


def _extract_products_from_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    products: List[Dict[str, Any]] = []
    seen = set()
    current: Dict[str, Any] | None = None

    for doc in docs:
        for line in str(doc.get("content", "")).splitlines():
            stripped = line.strip()
            if stripped.startswith("## "):
                name = stripped[3:].strip()
                current = {"name": name}
                if name not in seen:
                    products.append(current)
                    seen.add(name)
            elif current and stripped.startswith("- 品类:"):
                current["category"] = stripped.split(":", 1)[1].strip()
            elif current and stripped.startswith("- 价格:"):
                raw = stripped.split(":", 1)[1].strip().replace("元", "")
                if raw.isdigit():
                    current["price"] = int(raw)
            elif current and stripped.startswith("- 亮点:"):
                current["highlights"] = [
                    item.strip() for item in stripped.split(":", 1)[1].split(",") if item.strip()
                ]
    return [item for item in products if item.get("name")]


def _get_products_from_state(state: AgentState) -> List[Dict[str, Any]]:
    extraction_context = state.get("extraction_context") or {}
    products = extraction_context.get("products") or []
    if products:
        return products

    for item in reversed(state.get("step_results") or []):
        if item.get("worker") != "extraction_agent":
            continue
        artifacts = item.get("artifacts") or {}
        products = artifacts.get("products") or []
        if products:
            return products

    return _extract_products_from_docs(state.get("retrieved_docs", []))


def recommendation_agent_node(state: AgentState) -> dict:
    question = state["question"]
    budget = _parse_budget(question)
    category = _infer_category(question)
    preferences = _infer_preferences(question)
    scenarios = _infer_scenarios(question)
    products = _get_products_from_state(state)
    filtered_payload = catalog_filter.invoke(
        {
            "products": products,
            "budget": budget,
            "category": category,
            "preferences": preferences,
            "scenarios": scenarios,
        }
    )
    filtered_products = filtered_payload.get("products", [])
    hard_budget_products = filtered_payload.get("hard_budget_products", [])
    tool_results = [{"tool": "catalog_filter", "result": filtered_payload}]

    ranked = []
    ranking_pool = hard_budget_products or filtered_products
    if ranking_pool:
        ranked_payload = candidate_ranker.invoke(
            {
                "products": ranking_pool,
                "budget": budget,
                "category": category,
                "preferences": preferences,
                "scenarios": scenarios,
            }
        )
        ranked = ranked_payload.get("ranked_products", [])
        tool_results.append({"tool": "candidate_ranker", "result": ranked_payload})
    top_candidates = ranked[:2]
    gaps = []
    if budget is not None and not hard_budget_products:
        gaps.append("没有严格预算内候选，已放宽到接近预算范围")
    if category and not any((item.get("category") == category) for item in products):
        gaps.append(f"未找到品类为{category}的候选")
    if preferences and not any(item.get("matched_preferences") for item in ranked):
        gaps.append("候选中未明显命中偏好亮点")

    status = "success" if top_candidates else "partial"
    normalized_output = build_worker_output(
        worker="recommendation_agent",
        status=status,
        summary="已生成推荐候选" if top_candidates else "未找到满足条件的推荐候选",
        artifacts={
            "recommendations": top_candidates,
            "filtered_candidates": filtered_products,
            "excluded_candidates": filtered_payload.get("excluded_candidates", []),
            "decision_basis": {
                "budget": budget,
                "category": category,
                "preferences": preferences,
                "scenarios": scenarios,
            },
            "selection_summary": {
                **(filtered_payload.get("filter_summary") or {}),
                "ranking_pool_count": len(ranking_pool),
                "recommendation_count": len(top_candidates),
            },
            "coverage_gaps": gaps,
            "tool_results": [
                *tool_results
            ],
        },
        signals=["recommendation_ready"] if top_candidates else [],
        confidence=0.82 if top_candidates else 0.25,
    )

    next_step = "judge" if state.get("execution_plan") else "response_agent"
    return {
        "next_step": next_step,
        "active_agent": "recommendation_agent",
        "recommendation_context": normalized_output["artifacts"],
        "tool_results": tool_results,
        "last_worker_output": normalized_output,
        "agent_outputs": [
            {
                "agent": "recommendation_agent",
                "recommendation_count": len(top_candidates),
                "status": status,
            }
        ],
    }
