"""Optional rerank tool for retrieval results."""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.tools import tool


def _rerank_score(query: str, doc: dict[str, Any]) -> float:
    meta = doc.get("metadata", {})
    base = float(meta.get("rerank_score", 0) or 0) + float(meta.get("rrf_score", 0) or 0)
    content = str(doc.get("content", ""))
    query_terms = [term for term in query.replace("，", " ").replace(",", " ").split() if term]
    lexical_bonus = sum(1.0 for term in query_terms if term and term in content)
    file_name = str(meta.get("file_name") or meta.get("source") or "").lower()
    recommendation_signals = ["推荐", "买哪个", "怎么选", "适合", "预算", "对比"]
    guide_bonus = (
        1.5
        if ("guides" in file_name or "shopping_guide" in file_name)
        and any(signal in query for signal in recommendation_signals)
        else 0.0
    )
    return base + lexical_bonus + guide_bonus


@tool
def rerank_tool(
    query: Annotated[str, "用户查询"],
    docs: Annotated[list[dict[str, Any]], "候选文档列表"],
    top_k: Annotated[int, "保留文档数"] = 8,
) -> dict[str, Any]:
    """Rerank retrieved docs with lightweight lexical and metadata signals."""

    scored = []
    for doc in docs or []:
        score = _rerank_score(query, doc)
        updated = dict(doc)
        metadata = dict(updated.get("metadata", {}))
        metadata["tool_rerank_score"] = round(score, 4)
        updated["metadata"] = metadata
        scored.append(updated)

    scored.sort(key=lambda item: item.get("metadata", {}).get("tool_rerank_score", 0), reverse=True)
    return {"docs": scored[:top_k]}
