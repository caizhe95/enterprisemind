"""Section-aware matching helpers for metadata-rich Markdown docs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from graph.agents.field_utils import normalize_candidate_field


_SCHEMA_CACHE: dict[str, Any] | None = None


def _load_section_schema() -> dict[str, Any]:
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is not None:
        return _SCHEMA_CACHE

    default_schema = {
        "section_suffixes": [
            "关键点",
            "要点",
            "说明",
            "内容",
            "规则",
            "要求",
            "详情",
            "信息",
        ],
        "section_synonym_groups": [
            ["退换货", "退货", "换货"],
            ["保修", "质保"],
            ["发票", "开票"],
            ["配送", "物流", "发货"],
            ["会员", "vip"],
        ],
    }

    schema_path = Path(__file__).resolve().parents[2] / "config" / "section_schema.json"
    try:
        data = json.loads(schema_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            _SCHEMA_CACHE = {
                "section_suffixes": list(data.get("section_suffixes") or default_schema["section_suffixes"]),
                "section_synonym_groups": [
                    list(group)
                    for group in (data.get("section_synonym_groups") or default_schema["section_synonym_groups"])
                    if isinstance(group, list) and group
                ],
            }
            return _SCHEMA_CACHE
    except Exception:
        pass

    _SCHEMA_CACHE = default_schema
    return _SCHEMA_CACHE


def get_section_suffixes() -> tuple[str, ...]:
    return tuple(_load_section_schema()["section_suffixes"])


def get_section_synonym_groups() -> tuple[tuple[str, ...], ...]:
    groups = _load_section_schema()["section_synonym_groups"]
    return tuple(tuple(group) for group in groups)


def normalize_section_text(text: str) -> str:
    normalized = re.sub(r"[：:，,。；;、\s]+", "", text or "")
    for suffix in get_section_suffixes():
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
    return normalized


def section_synonym_forms(text: str) -> list[str]:
    base = normalize_section_text(text)
    if not base:
        return []

    forms = {base}
    if not base.endswith("政策"):
        forms.add(f"{base}政策")

    for group in get_section_synonym_groups():
        if any(term in base for term in group):
            for term in group:
                forms.add(base.replace(next(g for g in group if g in base), term))
            expanded = set(forms)
            for item in expanded:
                if not item.endswith("政策"):
                    forms.add(f"{item}政策")

    return sorted(forms, key=len, reverse=True)


def infer_section_targets(query: str, requested_fields: list[str] | None = None) -> list[str]:
    targets: list[str] = []

    for field in requested_fields or []:
        targets.extend(section_synonym_forms(field))

    single = normalize_candidate_field(query)
    if single:
        targets.extend(section_synonym_forms(single))

    compact_query = normalize_section_text(query)
    for group in get_section_synonym_groups():
        if any(term in compact_query for term in group):
            targets.extend(section_synonym_forms(next(term for term in group if term in compact_query)))

    deduped: list[str] = []
    seen = set()
    for target in targets:
        if target and target not in seen:
            seen.add(target)
            deduped.append(target)
    return deduped[:8]


def build_section_metadata_text(metadata: dict[str, Any]) -> str:
    return " ".join(
        str(metadata.get(key, "")) for key in ("H1", "H2", "H3", "header_path", "source_name", "file_name")
    ).strip()


def section_match_score(targets: list[str], content: str, metadata: dict[str, Any]) -> float:
    if not targets:
        return 0.0

    metadata_text = normalize_section_text(build_section_metadata_text(metadata))
    body_text = normalize_section_text(content)
    score = 0.0
    for target in targets:
        normalized_target = normalize_section_text(target)
        if not normalized_target:
            continue
        if normalized_target in metadata_text:
            score += 2.0
        elif metadata_text in normalized_target and metadata_text:
            score += 1.5
        if normalized_target in body_text:
            score += 0.8
        elif body_text and body_text[:40] and normalized_target in body_text[:80]:
            score += 0.4
    return score
