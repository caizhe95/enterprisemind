"""Rank filtered product candidates for recommendation."""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.tools import tool


def _score_product(
    product: dict[str, Any],
    budget: int | None,
    category: str | None,
    preferences: list[str],
    scenarios: list[str],
) -> dict[str, Any]:
    score = 0.0
    reasons: list[str] = []
    price = product.get("price")
    highlights = product.get("highlights") or []
    matched_preferences: list[str] = []
    matched_scenarios: list[str] = []
    budget_status = "unknown"

    if category and product.get("category") == category:
        score += 3.0
        reasons.append(f"属于{category}")

    if budget is not None and price is not None:
        if price <= budget:
            score += 3.0
            reasons.append("价格在预算内")
            budget_status = "within_budget"
        elif price <= budget * 1.15:
            score += 1.0
            reasons.append("价格接近预算")
            budget_status = "near_budget"
        else:
            score -= 2.0
            budget_status = "over_budget"
    elif price is not None:
        budget_status = "budget_not_provided"

    for pref in preferences:
        if any(pref in item for item in highlights):
            score += 1.5
            reasons.append(f"{pref}表现更匹配")
            matched_preferences.append(pref)

    if "性价比" in preferences and price is not None and budget is not None and price <= budget:
        score += 1.0

    scenario_weights = {
        "办公": {"长续航": 1.5, "轻薄便携": 1.5, "屏幕素质高": 1.0},
        "学生": {"学习友好": 2.0, "护眼屏": 1.5, "轻便易携": 1.2, "长续航": 1.0},
        "游戏": {"高性能": 2.2, "散热稳定": 1.8, "屏幕素质高": 1.0},
        "拍照": {"影像稳定": 2.0, "屏幕细腻": 1.0, "信号稳定": 0.5},
        "轻薄续航": {"轻薄便携": 2.0, "长续航": 2.0, "轻便易携": 1.5},
    }
    for scenario in scenarios:
        weights = scenario_weights.get(scenario, {})
        for highlight, bonus in weights.items():
            if any(highlight in item for item in highlights):
                score += bonus
                reasons.append(f"更适合{scenario}")
                matched_scenarios.append(scenario)

    return {
        "name": product["name"],
        "price": price,
        "category": product.get("category"),
        "highlights": highlights,
        "score": round(score, 2),
        "reasons": list(dict.fromkeys(reasons))[:3],
        "budget_status": budget_status,
        "matched_preferences": list(dict.fromkeys(matched_preferences)),
        "matched_scenarios": list(dict.fromkeys(matched_scenarios)),
    }


@tool
def candidate_ranker(
    products: Annotated[list[dict[str, Any]], "已过滤的候选商品"],
    budget: Annotated[int | None, "预算"] = None,
    category: Annotated[str | None, "品类"] = None,
    preferences: Annotated[list[str], "偏好关键词"] = [],
    scenarios: Annotated[list[str], "场景标签"] = [],
) -> dict[str, Any]:
    """Rank candidate products and generate recommendation reasons."""

    ranked = [
        _score_product(
            product,
            budget=budget,
            category=category,
            preferences=preferences,
            scenarios=scenarios,
        )
        for product in products or []
    ]
    ranked = [item for item in ranked if item["score"] > 0]
    budget_priority = {"within_budget": 2, "near_budget": 1, "budget_not_provided": 1, "unknown": 1, "over_budget": 0}
    ranked.sort(
        key=lambda item: (
            budget_priority.get(item.get("budget_status", "unknown"), 0),
            item["score"],
            -(item.get("price") or 10**9),
        ),
        reverse=True,
    )
    return {"ranked_products": ranked}
