"""Structured extraction tool for product docs."""

from __future__ import annotations

import re
from typing import Annotated, Any

from langchain_core.tools import tool

from graph.agents.field_utils import (
    FIELD_MARKERS_PATTERN,
    field_aliases_for,
    extract_fields_by_text,
    normalize_candidate_field,
)
from graph.agents.section_utils import (
    build_section_metadata_text,
    infer_section_targets,
    section_match_score,
)


def _extract_requested_fields(query: str) -> list[str]:
    fields = extract_fields_by_text(query)
    if fields:
        return fields

    section_match = re.search(
        r"([\u4e00-\u9fffA-Za-z0-9]{1,12}(?:政策|规则|要求|说明|关键点|要点))",
        query,
    )
    if section_match:
        candidate = section_match.group(1)
        if "的" in candidate:
            candidate = candidate.split("的")[-1].strip()
        return [candidate]

    single_candidate = normalize_candidate_field(query)
    if single_candidate:
        return [single_candidate]

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


def _extract_query_entity(query: str) -> str:
    text = re.sub(r"[？?。！!]", "", query).strip()
    for marker in [
        "起售价",
        "售价",
        "价格",
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
    text = re.sub(r"(请问|帮我|查下|查一下)$", "", text).strip()
    text = re.sub(r"的+$", "", text).strip()
    return text


def _extract_metric_value_for_entity(
    query: str, docs: list[dict[str, Any]], metric: str
) -> int | None:
    entity = _extract_query_entity(query)
    if not entity:
        return None

    pattern = {
        "价格": r"(?:价格|售价)\s*[:：]\s*(\d+)\s*元?",
        "销量": r"销量\s*[:：]?\s*(\d+)",
        "销售额": r"销售额\s*[:：]?\s*(\d+)",
        "续航": r"续航\s*[:：]?\s*(\d+)",
    }.get(metric, rf"{re.escape(metric)}\s*[:：]?\s*(\d+)")

    for doc in docs:
        content = str(doc.get("content", ""))
        metadata = doc.get("metadata", {}) or {}
        metadata_text = build_section_metadata_text(metadata)
        if entity not in content and entity not in metadata_text:
            continue
        if f"## {entity}" in content:
            start = content.find(f"## {entity}")
            next_header = content.find("\n## ", start + 1)
            block = content[start:] if next_header == -1 else content[start:next_header]
            match = re.search(pattern, block)
            if match:
                return int(match.group(1))
        match = re.search(pattern, content)
        if match:
            return int(match.group(1))
    return None


def _filter_docs_by_entity(query: str, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entity = _extract_query_entity(query)
    if not entity:
        return docs

    def _entity_stem(text: str) -> str:
        cleaned = re.sub(r"\d+\s*代", "", text)
        return re.sub(r"\s+", "", cleaned).strip()

    matched = []
    query_generation = re.search(r"(\d+)\s*代", entity)
    query_generation = query_generation.group(1) if query_generation else None
    query_stem = _entity_stem(entity)
    for doc in docs:
        content = str(doc.get("content", ""))
        metadata = doc.get("metadata", {}) or {}
        metadata_text = build_section_metadata_text(metadata)
        if entity in content:
            matched.append(doc)
            continue
        if entity in metadata_text:
            matched.append(doc)
            continue
        header_match = re.search(r"##\s*([^\n]+)", content)
        if not header_match:
            continue
        header = header_match.group(1).strip()
        if query_stem and query_stem not in _entity_stem(header):
            continue
        header_generation = re.search(r"(\d+)\s*代", header)
        header_generation = header_generation.group(1) if header_generation else None
        if query_generation and header_generation and header_generation != query_generation:
            continue
        matched.append(doc)
    return matched or docs


def _extract_text_fields(fields: list[str], docs: list[dict[str, Any]]) -> dict[str, str]:
    extracted: dict[str, str] = {}
    for field in fields:
        section_targets = infer_section_targets(field, [field])
        aliases = field_aliases_for(field)
        for doc in docs:
            content = str(doc.get("content", ""))
            metadata = doc.get("metadata", {}) or {}
            for alias in aliases:
                match = re.search(rf"{re.escape(alias)}\s*[:：]\s*([^\n]+)", content)
                if match:
                    extracted[field] = match.group(1).strip(" -：:")
                    break
                record_match = re.search(rf"{re.escape(alias)}=([^;；\n]+)", content)
                if record_match:
                    extracted[field] = record_match.group(1).strip(" -：:")
                    break
            if field in extracted:
                break
            section_value = _extract_section_list_item(section_targets, content, metadata)
            if section_value:
                extracted[field] = section_value
                break
    return extracted


def _extract_section_list_item(
    section_targets: list[str], content: str, metadata: dict[str, Any]
) -> str | None:
    if section_match_score(section_targets, content, metadata) <= 0:
        return None

    numbered_lines = []
    for line in content.splitlines():
        stripped = line.strip()
        cleaned = re.sub(r"^\d+\.\s*", "", stripped)
        if cleaned and cleaned != stripped:
            numbered_lines.append(cleaned)

    if not numbered_lines:
        return None

    return numbered_lines[0]


def _extract_sales_record_price(query: str, docs: list[dict[str, Any]]) -> int | None:
    entity_docs = _filter_docs_by_entity(query, docs)
    for doc in entity_docs:
        content = str(doc.get("content", ""))
        for line in content.splitlines():
            if "销售额=" not in line:
                continue
            amount_match = re.search(r"销售额=([0-9]+)", line)
            qty_match = re.search(r"(?:销量|数量)=([0-9]+)", line)
            if not amount_match or not qty_match:
                continue
            amount = int(amount_match.group(1))
            qty = int(qty_match.group(1))
            if qty > 0:
                return round(amount / qty)
    return None


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

    docs = _filter_docs_by_entity(query, docs)
    entity = _extract_query_entity(query)
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
        value = _extract_metric_value_for_entity(query, docs, target)
        if value is None and not entity:
            value = _extract_metric_value(docs, target)
        if value is None and target == "价格":
            value = _extract_sales_record_price(query, docs)
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
