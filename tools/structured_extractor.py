"""Structured extraction tool for product docs."""

from __future__ import annotations

import re
from typing import Annotated, Any

from langchain_core.tools import tool

from graph.agents.field_utils import FIELD_MARKERS_PATTERN, extract_fields_by_text


def _extract_requested_fields(query: str) -> list[str]:
    fields = extract_fields_by_text(query)
    if fields:
        return fields

    matches = re.findall(FIELD_MARKERS_PATTERN, query)
    cleaned = list(dict.fromkeys([match.strip() for match in matches if match.strip()]))
    return cleaned[:4]


def _extract_metric_value(docs: list[dict[str, Any]], metric: str) -> int | None:
    patterns = {
        "价格": r"(?:价格|售价)\s*[:：]\s*(\d+)\s*元?",
        "销量": r"销量\s*[:：]?\s*(\d+)",
        "销售额": r"销售额\s*[:：]?\s*(\d+)",
        "续航": r"续航\s*[:：]?\s*(\d+)",
    }
    pattern = patterns.get(metric, rf"{re.escape(metric)}\s*[:：]?\s*(\d+)")
    for doc in docs:
        match = re.search(pattern, str(doc.get("content", "")))
        if match:
            return int(match.group(1))
    return None


def _extract_text_fields(fields: list[str], docs: list[dict[str, Any]]) -> dict[str, str]:
    extracted: dict[str, str] = {}
    for field in fields:
        for doc in docs:
            content = str(doc.get("content", ""))
            match = re.search(rf"{re.escape(field)}\s*[:：]\s*([^\n]+)", content)
            if match:
                extracted[field] = match.group(1).strip(" -：:")
                break
    return extracted


def _extract_products_from_docs(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    products: list[dict[str, Any]] = []
    seen = set()
    current: dict[str, Any] | None = None

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
                    item.strip()
                    for item in stripped.split(":", 1)[1].split(",")
                    if item.strip()
                ]

    return [item for item in products if item.get("name")]


@tool
def structured_extractor(
    query: Annotated[str, "原始问题或当前worker输入"],
    docs: Annotated[list[dict[str, Any]], "待抽取的文档列表"],
    metric: Annotated[str | None, "可选比较指标"] = None,
) -> dict[str, Any]:
    """Extract fields, metrics, and product candidates from retrieved docs."""

    requested_fields = _extract_requested_fields(query)
    fields = _extract_text_fields(requested_fields, docs)
    metrics: dict[str, dict[str, Any]] = {}

    target_metrics = [metric] if metric else []
    for item in requested_fields:
        if item in {"价格", "销量", "销售额", "续航"} and item not in target_metrics:
            target_metrics.append(item)

    for target in target_metrics:
        if not target:
            continue
        value = _extract_metric_value(docs, target)
        if value is not None:
            metrics[target] = {"value": value, "unit": "元" if target == "价格" else None}

    products = _extract_products_from_docs(docs)
    return {
        "fields": fields,
        "metrics": metrics,
        "products": products,
        "requested_fields": requested_fields,
        "doc_count": len(docs),
    }
