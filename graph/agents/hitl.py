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

    mapping = {
        "sql": "sql_agent",
        "search": "search_agent",
        "calculation": "calculation_agent",
        "retrieval": "retrieval_agent",
    }
    next_step = mapping[chosen]
    decision_log = f"HITL策略确认: chosen={chosen}, recommended={recommended}"

    return {
        "routing_hint": chosen,
        "hitl_request": None,
        "next_step": next_step,
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
            "next_step": "search_agent",
            "final_answer": None,
            "citations": [],
        }

    retry_question = f"{question}\n请基于现有证据保守回答，证据不足时明确说无法确定。"
    return {
        **base,
        "question": retry_question,
        "routing_hint": "retrieval",
        "next_step": "retrieval_agent",
        "final_answer": None,
        "citations": [],
    }
