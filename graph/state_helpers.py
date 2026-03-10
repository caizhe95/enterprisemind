"""Shared helpers for graph state initialization and interrupt parsing."""

from __future__ import annotations

from typing import Any, Optional

from graph.state import AgentState


def build_initial_state(
    question: str,
    session_id: str,
    user_id: Optional[str],
    routing_hint: Optional[str] = None,
) -> AgentState:
    return {
        "messages": [],
        "question": question,
        "question_type": None,
        "routing_hint": routing_hint,
        "retry_policy": None,
        "session_id": session_id,
        "user_id": user_id,
        "hitl_request": None,
        "shopping_profile": {},
        "retrieval_grade": None,
        "self_rag_eval": None,
        "reflection_count": 0,
        "thought": None,
        "action": None,
        "action_input": None,
        "observation": None,
        "retrieved_docs": [],
        "retrieval_count": 0,
        "tool_results": [],
        "generated_sql": None,
        "sql_result": None,
        "active_agent": "supervisor",
        "supervisor_decision": None,
        "agent_outputs": [],
        "execution_plan": [],
        "plan_version": 0,
        "current_step_index": 0,
        "worker_input": None,
        "last_worker_output": None,
        "step_results": [],
        "extraction_context": None,
        "calculation_expression": None,
        "comparison_context": None,
        "recommendation_context": None,
        "step_retry_counts": {},
        "replan_count": 0,
        "replan_reason": None,
        "next_step": "supervisor",
        "final_answer": None,
        "citations": [],
        "working_memory": [],
        "token_usage": 0,
        "execution_trace": [],
        "worker_trace": [],
        "tool_calls": [],
        "step_count": 0,
        "guardrail_result": None,
    }


def normalize_interrupt(raw_interrupt: Any) -> dict[str, Any]:
    value = raw_interrupt
    if isinstance(raw_interrupt, (list, tuple)) and raw_interrupt:
        first = raw_interrupt[0]
        value = getattr(first, "value", first)
    elif hasattr(raw_interrupt, "value"):
        value = raw_interrupt.value

    return value if isinstance(value, dict) else {"type": "unknown", "raw": value}
