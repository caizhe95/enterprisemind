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
        "routing_hint": routing_hint,
        "retry_policy": None,
        "session_id": session_id,
        "user_id": user_id,
        "hitl_request": None,
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
        "next_step": "supervisor",
        "final_answer": None,
        "citations": [],
        "working_memory": [],
        "token_usage": 0,
        "execution_trace": [],
    }


def normalize_interrupt(raw_interrupt: Any) -> dict[str, Any]:
    value = raw_interrupt
    if isinstance(raw_interrupt, (list, tuple)) and raw_interrupt:
        first = raw_interrupt[0]
        value = getattr(first, "value", first)
    elif hasattr(raw_interrupt, "value"):
        value = raw_interrupt.value

    return value if isinstance(value, dict) else {"type": "unknown", "raw": value}
