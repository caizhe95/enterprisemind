"""Normalize extracted fields, metrics, and product payloads."""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.tools import tool


FIELD_ALIASES = {
    "售价": "价格",
    "价钱": "价格",
    "费用": "价格",
}


def _canonical_field(name: str) -> str:
    return FIELD_ALIASES.get(name, name)


def _normalize_metric_value(field: str, payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        value = payload.get("value")
        unit = payload.get("unit")
    else:
        value = payload
        unit = None

    if field == "价格":
        if unit in {None, ""}:
            unit = "元"
    return {"value": value, "unit": unit}


@tool
def field_normalizer(
    fields: Annotated[dict[str, Any], "抽取得到的字段映射"],
    metrics: Annotated[dict[str, Any], "抽取得到的指标映射"],
    products: Annotated[list[dict[str, Any]], "抽取得到的商品候选"],
) -> dict[str, Any]:
    """Normalize field names, metric units, and product defaults."""

    normalized_fields: dict[str, str] = {}
    for key, value in (fields or {}).items():
        field = _canonical_field(str(key).strip())
        text = str(value).strip()
        if field == "价格" and text and not text.endswith("元") and text.isdigit():
            text = f"{text}元"
        normalized_fields[field] = text

    normalized_metrics: dict[str, dict[str, Any]] = {}
    for key, value in (metrics or {}).items():
        field = _canonical_field(str(key).strip())
        normalized_metrics[field] = _normalize_metric_value(field, value)

    normalized_products: list[dict[str, Any]] = []
    for product in products or []:
        item = dict(product)
        if item.get("price") is not None:
            try:
                item["price"] = int(item["price"])
            except (TypeError, ValueError):
                pass
        item["category"] = item.get("category")
        item["highlights"] = list(item.get("highlights") or [])
        normalized_products.append(item)

    return {
        "fields": normalized_fields,
        "metrics": normalized_metrics,
        "products": normalized_products,
    }
