"""Validation helpers for local product knowledge records."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def validate_product_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate product records used by retrieval and recommendation flows."""
    duplicate_names = []
    missing_required_fields = []
    grouped_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    required_fields = ["name", "SKU", "品类", "品牌", "价格"]

    for index, record in enumerate(records):
        name = _normalize_text(record.get("name"))
        if name:
            grouped_by_name[name].append(record)

        missing = [
            field for field in required_fields if not _normalize_text(record.get(field))
        ]
        if missing:
            missing_required_fields.append(
                {
                    "index": index,
                    "name": name or f"record_{index}",
                    "missing_fields": missing,
                }
            )

    for name, group in grouped_by_name.items():
        if len(group) < 2:
            continue
        duplicate_names.append(
            {
                "name": name,
                "count": len(group),
                "skus": sorted(
                    {
                        _normalize_text(item.get("SKU"))
                        for item in group
                        if _normalize_text(item.get("SKU"))
                    }
                ),
                "prices": sorted(
                    {
                        _normalize_text(item.get("价格"))
                        for item in group
                        if _normalize_text(item.get("价格"))
                    }
                ),
            }
        )

    issues = {
        "duplicate_names": sorted(duplicate_names, key=lambda item: item["name"]),
        "missing_required_fields": missing_required_fields,
    }
    has_issues = any(issues.values())
    return {
        "summary": {
            "total_records": len(records),
            "has_issues": has_issues,
            "duplicate_name_count": len(issues["duplicate_names"]),
            "missing_required_field_count": len(issues["missing_required_fields"]),
        },
        "issues": issues,
    }
