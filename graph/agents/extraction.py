"""Extraction worker for structuring retrieved docs."""

from __future__ import annotations

from graph.state import AgentState
from graph.agents.worker_contract import build_worker_output
from tools.field_normalizer import field_normalizer
from tools.structured_extractor import structured_extractor


def _current_metric(state: AgentState) -> str | None:
    plan = state.get("execution_plan") or []
    idx = int(state.get("current_step_index", 0) or 0)
    if not plan or idx >= len(plan):
        return None
    return plan[idx].get("metric")


def extraction_agent_node(state: AgentState) -> dict:
    """Extraction agent: parse structured fields and metrics from retrieved docs."""
    query = state.get("worker_input") or state["question"]
    docs = state.get("retrieved_docs", [])
    metric = _current_metric(state)
    extracted = structured_extractor.invoke(
        {
            "query": query,
            "docs": docs,
            "metric": metric,
        }
    )
    tool_results = [{"tool": "structured_extractor", "result": extracted}]

    needs_normalization = bool(extracted.get("fields") or extracted.get("metrics") or extracted.get("products"))
    if needs_normalization:
        normalized = field_normalizer.invoke(
            {
                "fields": extracted.get("fields", {}),
                "metrics": extracted.get("metrics", {}),
                "products": extracted.get("products", []),
            }
        )
        extracted = {**extracted, **normalized}
        tool_results.append({"tool": "field_normalizer", "result": normalized})

    signals = []
    if docs:
        signals.append("documents_found")
    if extracted.get("fields"):
        signals.append("field_values_found")
    if extracted.get("metrics"):
        signals.append("value_found")
    if extracted.get("products"):
        signals.append("candidate_products_found")
    if extracted.get("fields") or extracted.get("metrics") or extracted.get("products"):
        signals.append("structured_data_ready")

    status = "success" if docs else "partial"
    normalized_output = build_worker_output(
        worker="extraction_agent",
        status=status,
        summary="结构化抽取完成" if docs else "无可抽取文档",
        artifacts=extracted,
        signals=signals,
        confidence=0.86 if docs else 0.2,
    )

    next_step = "judge" if state.get("execution_plan") else "response_agent"
    return {
        "next_step": next_step,
        "active_agent": "extraction_agent",
        "extraction_context": extracted,
        "tool_results": tool_results,
        "last_worker_output": normalized_output,
        "agent_outputs": [
            {
                "agent": "extraction_agent",
                "field_count": len(extracted.get("fields", {})),
                "metric_count": len(extracted.get("metrics", {})),
                "product_count": len(extracted.get("products", [])),
                "normalized": needs_normalization,
                "status": status,
            }
        ],
    }
