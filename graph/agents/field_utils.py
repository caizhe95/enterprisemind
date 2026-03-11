"""Shared field extraction helpers for shopping QA."""

from __future__ import annotations

import json
import re
from pathlib import Path

from graph.agents.common import GENERIC_FIELD_MARKERS, _looks_like_named_entity


_FIELD_SCHEMA_CACHE: dict | None = None


def _load_field_schema() -> dict:
    global _FIELD_SCHEMA_CACHE
    if _FIELD_SCHEMA_CACHE is not None:
        return _FIELD_SCHEMA_CACHE

    default_schema = {
        "field_alias_groups": [
            ["价格", "售价", "价钱", "费用", "多少钱"],
            ["品类", "类别", "分类"],
            ["参数", "配置", "规格"],
            ["发布时间", "上市时间"],
            ["保修", "质保"],
            ["销量", "销售量"],
            ["销售额", "成交额"],
        ],
        "metric_fields": ["价格", "销量", "销售额", "续航"],
        "query_synonym_groups": [
            ["销售额", "营收", "收入", "销售金额", "营业额"],
            ["产品", "商品", "货物", "SKU", "型号", "Item"],
            ["客户", "用户", "顾客", "购买者", "消费者", "买家"],
            ["价格", "单价", "售价", "定价", "多少钱", "费用"],
            ["库存", "存货", "仓储", "备货量", "存储量"],
            ["优惠", "折扣", "促销", "活动价", "减免", "让利"],
            ["排名", "排序", "前几名", "Top", "最佳", "最受欢迎"],
            ["保修", "质保", "维修", "售后保障", "保修政策"],
            ["发票", "电子发票", "纸质发票", "开票", "发票抬头"],
            ["配送", "发货", "物流", "到货", "配送时效"],
            ["会员", "会员等级", "会员权益", "折扣等级", "会员规则"],
            ["渠道", "线上", "线下", "销售渠道", "购买渠道"],
            ["销量", "销售量", "卖出数量", "出货量", "成交量"],
            ["退换货", "退货", "换货", "售后政策", "退换政策"],
            ["价保", "价格保护", "保价", "补差", "价保政策"],
            ["活动", "促销活动", "优惠活动", "限时活动", "大促"],
        ],
    }

    schema_path = Path(__file__).resolve().parents[2] / "config" / "field_schema.json"
    try:
        data = json.loads(schema_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            _FIELD_SCHEMA_CACHE = {
                "field_alias_groups": [
                    list(group)
                    for group in (data.get("field_alias_groups") or default_schema["field_alias_groups"])
                    if isinstance(group, list) and group
                ],
                "metric_fields": list(data.get("metric_fields") or default_schema["metric_fields"]),
                "query_synonym_groups": [
                    list(group)
                    for group in (
                        data.get("query_synonym_groups")
                        or default_schema["query_synonym_groups"]
                    )
                    if isinstance(group, list) and group
                ],
            }
            return _FIELD_SCHEMA_CACHE
    except Exception:
        pass

    _FIELD_SCHEMA_CACHE = default_schema
    return _FIELD_SCHEMA_CACHE


def get_field_alias_groups() -> list[list[str]]:
    return _load_field_schema()["field_alias_groups"]


def get_metric_fields() -> list[str]:
    return _load_field_schema()["metric_fields"]


def get_query_synonym_groups() -> list[list[str]]:
    return _load_field_schema()["query_synonym_groups"]


def field_aliases_for(name: str) -> list[str]:
    candidate = (name or "").strip()
    canonical = canonicalize_field_name(candidate)
    aliases = [candidate] if candidate else []
    for group in get_field_alias_groups():
        if canonical in group or candidate in group:
            aliases.extend(group)
            break
    deduped: list[str] = []
    seen = set()
    for item in aliases:
        if item and item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def canonicalize_field_name(name: str) -> str:
    candidate = (name or "").strip()
    if not candidate:
        return ""
    for group in get_field_alias_groups():
        if candidate in group:
            return group[0]
    return candidate


FIELD_MARKERS_PATTERN = "|".join(
    sorted(
        {
            re.escape(marker)
            for marker in (
                GENERIC_FIELD_MARKERS
                + ["品类"]
                + [alias for group in get_field_alias_groups() for alias in group]
            )
        },
        key=len,
        reverse=True,
    )
)


def has_explicit_field_list_signal(question: str) -> bool:
    return any(
        marker in question
        for marker in ["分别", "各自", "分别是", "分别为", "分别是什么", "各是多少"]
    )


def is_placeholder_field(field: str) -> bool:
    s = (field or "").strip().lower()
    return bool(re.match(r"^(字段\d*|field\d*|item\d*)$", s))


def normalize_candidate_field(text: str) -> str:
    candidate = text.strip("：: ，,。；; ")
    if not candidate:
        return ""

    marker_match = re.search(rf"({FIELD_MARKERS_PATTERN})$", candidate)
    if marker_match:
        return canonicalize_field_name(marker_match.group(1))

    if "的" in candidate:
        candidate = candidate.split("的")[-1].strip()
        marker_match = re.search(rf"({FIELD_MARKERS_PATTERN})$", candidate)
        if marker_match:
            return canonicalize_field_name(marker_match.group(1))

    if _looks_like_named_entity(candidate):
        return ""

    if re.search(r"(哪个|哪款|哪一个|谁|是否|能否|怎么|如何|为什么|多少|几|多大|更|比|差)", candidate):
        return ""

    if len(candidate) > 12:
        return ""

    return canonicalize_field_name(candidate)


def extract_fields_by_text(question: str) -> list[str]:
    text = re.sub(r"[？?。！!]", "", question).strip()
    for marker in [
        "分别是什么",
        "分别是",
        "分别为",
        "分别",
        "各自",
        "各是多少",
        "是多少",
    ]:
        if marker in text:
            text = text.split(marker)[0].strip()
            break

    parts = re.split(r"[、,，/]|以及|和|及|与", text)
    parts = [p.strip() for p in parts if p.strip()]
    fields = []
    for p in parts:
        c = normalize_candidate_field(p)
        if not c or is_placeholder_field(c):
            continue
        fields.append(c)
    fields = list(dict.fromkeys(fields))
    return fields[:8] if len(fields) >= 2 else []
