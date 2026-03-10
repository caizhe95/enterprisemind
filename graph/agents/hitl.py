"""HITL interrupt 节点。"""

from langgraph.types import interrupt

from graph.state import AgentState


def hitl_strategy_confirm_node(state: AgentState) -> dict:
    """模糊路由确认：等待用户选择策略后恢复执行。"""
    request = state.get("hitl_request") or {}
    decision = interrupt(request)

    recommended = request.get("recommended", "retrieval")
    chosen = decision
    if isinstance(decision, dict):
        chosen = decision.get("strategy")
    chosen = (chosen or recommended).strip().lower()
    if chosen == "auto":
        chosen = recommended
    if chosen not in {"sql", "search", "calculation", "retrieval"}:
        chosen = recommended

    decision_log = f"HITL策略确认: chosen={chosen}, recommended={recommended}"

    return {
        "routing_hint": chosen,
        "hitl_request": None,
        "next_step": "planner",
        "execution_trace": [
            {"node": "hitl_strategy_confirm", "decision": decision_log}
        ],
    }


def hitl_low_conf_confirm_node(state: AgentState) -> dict:
    """低置信答案确认：接受或触发重试路径。"""
    request = state.get("hitl_request") or {}
    decision = interrupt(request)

    action = decision
    if isinstance(decision, dict):
        action = decision.get("action")
    action = (action or "accept").strip().lower()
    if action not in {"accept", "web_retry", "conservative_retry"}:
        action = "accept"

    base = {
        "retry_policy": action,
        "hitl_request": None,
        "execution_trace": [
            {"node": "hitl_low_conf_confirm", "decision": f"action={action}"}
        ],
    }
    if action == "accept":
        return {**base, "next_step": "end"}

    question = state["question"]
    if action == "web_retry":
        retry_question = f"{question}\n请补充可验证来源；若不确定请明确说明。"
        return {
            **base,
            "question": retry_question,
            "routing_hint": "search",
            "retrieved_docs": [],
            "retrieval_grade": None,
            "tool_results": [],
            "tool_calls": [],
            "worker_trace": [],
            "step_results": [],
            "last_worker_output": None,
            "extraction_context": None,
            "comparison_context": None,
            "calculation_expression": None,
            "recommendation_context": None,
            "next_step": "planner",
            "final_answer": None,
            "citations": [],
        }

    retry_question = f"{question}\n请基于现有证据保守回答，证据不足时明确说无法确定。"
    return {
        **base,
        "question": retry_question,
        "routing_hint": "retrieval",
        "retrieved_docs": [],
        "retrieval_grade": None,
        "tool_results": [],
        "tool_calls": [],
        "worker_trace": [],
        "step_results": [],
        "last_worker_output": None,
        "extraction_context": None,
        "comparison_context": None,
        "calculation_expression": None,
        "recommendation_context": None,
        "next_step": "planner",
        "final_answer": None,
        "citations": [],
    }


def hitl_worker_confirm_node(state: AgentState) -> dict:
    """Worker 侧导购槽位确认。"""
    request = state.get("hitl_request") or {}
    decision = interrupt(request)

    slot_answer = ""
    if isinstance(decision, dict):
        slot_answer = str(decision.get("slot_answer") or "").strip()
    else:
        slot_answer = str(decision or "").strip()

    if not slot_answer:
        slot_answer = "预算不限，优先性价比"

    profile = dict(state.get("shopping_profile") or {})
    profile["user_preference"] = slot_answer
    question = state["question"]
    if "用户补充偏好" not in question:
        question = f"{question}\n用户补充偏好：{slot_answer}"

    return {
        "question": question,
        "shopping_profile": profile,
        "hitl_request": None,
        "next_step": "retrieval_agent",
        "execution_trace": [
            {
                "node": "hitl_worker_confirm",
                "decision": f"slot_answer={slot_answer}",
            }
        ],
    }
