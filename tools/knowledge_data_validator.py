"""Knowledge data validation helpers for product markdown sources."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any


REQUIRED_PRODUCT_FIELDS = ("SKU", "品类", "品牌", "价格", "上市日期", "保修", "亮点")


def parse_product_markdown(path: str | Path) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8")
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            if current:
                records.append(current)
            current = {"name": line[3:].strip()}
            continue
        if not current or not line.startswith("- "):
            continue
        body = line[2:]
        if ":" not in body and "：" not in body:
            continue
        parts = re.split(r"[:：]", body, maxsplit=1)
        if len(parts) != 2:
            continue
        key, value = parts
        current[key.strip()] = value.strip()

    if current:
        records.append(current)
    return records


def validate_product_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    issues: dict[str, list[dict[str, Any]]] = {
        "duplicate_names": [],
        "duplicate_skus": [],
        "missing_fields": [],
    }

    name_index: dict[str, list[dict[str, Any]]] = {}
    sku_index: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        name_index.setdefault(record.get("name", ""), []).append(record)
        sku = str(record.get("SKU", "")).strip()
        if sku:
            sku_index.setdefault(sku, []).append(record)

        missing = [field for field in REQUIRED_PRODUCT_FIELDS if not str(record.get(field, "")).strip()]
        if missing:
            issues["missing_fields"].append({"name": record.get("name"), "missing_fields": missing})

    for name, grouped in name_index.items():
        if name and len(grouped) > 1:
            prices = sorted({str(item.get("价格", "")).strip() for item in grouped if str(item.get("价格", "")).strip()})
            skus = sorted({str(item.get("SKU", "")).strip() for item in grouped if str(item.get("SKU", "")).strip()})
            issues["duplicate_names"].append(
                {"name": name, "count": len(grouped), "prices": prices, "skus": skus}
            )

    for sku, grouped in sku_index.items():
        if sku and len(grouped) > 1:
            issues["duplicate_skus"].append({"sku": sku, "count": len(grouped)})

    summary = {
        "record_count": len(records),
        "issue_count": sum(len(items) for items in issues.values()),
        "has_issues": any(bool(items) for items in issues.values()),
    }
    return {"summary": summary, "issues": issues}


def validate_product_markdown(path: str | Path) -> dict[str, Any]:
    records = parse_product_markdown(path)
    return validate_product_records(records)
