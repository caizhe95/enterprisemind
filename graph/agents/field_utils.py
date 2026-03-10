"""Shared field extraction helpers for shopping QA."""

from __future__ import annotations

import re

from graph.agents.common import GENERIC_FIELD_MARKERS, _looks_like_named_entity


FIELD_MARKERS_PATTERN = "|".join(
    sorted({re.escape(marker) for marker in GENERIC_FIELD_MARKERS + ["品类"]}, key=len, reverse=True)
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
        return marker_match.group(1)

    if "的" in candidate:
        candidate = candidate.split("的")[-1].strip()
        marker_match = re.search(rf"({FIELD_MARKERS_PATTERN})$", candidate)
        if marker_match:
            return marker_match.group(1)

    if _looks_like_named_entity(candidate):
        return ""

    if re.search(r"(哪个|哪款|哪一个|谁|是否|能否|怎么|如何|为什么|多少|几|多大|更|比|差)", candidate):
        return ""

    if len(candidate) > 12:
        return ""

    return candidate


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
