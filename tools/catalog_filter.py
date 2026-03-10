"""Candidate filtering tool for recommendation flow."""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.tools import tool


@tool
def catalog_filter(
    products: Annotated[list[dict[str, Any]], "商品候选列表"],
    budget: Annotated[int | None, "预算"] = None,
    category: Annotated[str | None, "品类"] = None,
    preferences: Annotated[list[str], "偏好关键词"] = [],
    scenarios: Annotated[list[str], "场景标签"] = [],
) -> dict[str, Any]:
    """Filter candidates by category, budget, preferences, and scenarios."""

    filtered = []
    for product in products:
        if category and product.get("category") != category:
            continue
        price = product.get("price")
        if budget is not None and price is not None and price > budget * 1.15:
            continue
        filtered.append(product)

    if not filtered:
        filtered = list(products)

    preferred = []
    for product in filtered:
        highlights = product.get("highlights") or []
        score = 0
        for pref in preferences:
            if any(pref in item for item in highlights):
                score += 1
        for scenario in scenarios:
            if any(scenario in item for item in highlights):
                score += 1
        preferred.append((score, product))

    preferred.sort(key=lambda item: item[0], reverse=True)
    ordered = [item[1] for item in preferred]

    return {
        "products": ordered or filtered,
        "applied_filters": {
            "budget": budget,
            "category": category,
            "preferences": preferences,
            "scenarios": scenarios,
        },
    }
