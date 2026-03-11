"""Supervisor与路由节点"""

from config import config
from graph.state import AgentState
from graph.agents.common import analyze_intent


def _should_confirm_ambiguity(analysis: dict) -> bool:
    rank = {"low": 1, "medium": 2, "high": 3}
    threshold = config.HITL_AMBIGUITY_CONFIDENCE_LEVEL
    if threshold not in rank:
        threshold = "medium"

    confidence = str(analysis.get("confidence", "medium")).lower()
    if confidence not in rank:
        confidence = "medium"

    by_confidence = rank[confidence] < rank[threshold]
    dual_route = bool(
        analysis.get("should_try_search") and analysis.get("should_try_retrieval")
    )
    return by_confidence or (config.HITL_REQUIRE_CONFIRM_ON_DUAL_ROUTE and dual_route)


def supervisor_node(state: AgentState) -> dict:
    """Supervisor：统一调度多Agent协同"""
    question = state["question"]
    routing_hint = state.get("routing_hint")
    if routing_hint in {"sql", "search", "calculation", "retrieval"}:
        routing = {
            "intent": routing_hint,
            "reason": "用户确认策略",
            "confidence": "high",
            "should_try_search": routing_hint == "search",
            "should_try_retrieval": routing_hint == "retrieval",
        }
    else:
        routing = analyze_intent(question)
        if routing.get("auto_route_to_search_on_dual") and routing.get("should_try_search"):
            routing = {
                **routing,
                "intent": "search",
                "reason": f"{routing['reason']}；公开商品事实题自动外部搜索",
                "confidence": "high",
                "should_try_retrieval": False,
            }
        if _should_confirm_ambiguity(routing):
            recommended = (
                "search"
                if routing.get("should_try_search")
                else routing.get("intent", "retrieval")
            )
            return {
                "next_step": "hitl_strategy_confirm",
                "active_agent": "supervisor",
                "hitl_request": {
                    "type": "strategy_confirm",
                    "question": question,
                    "analysis": routing,
                    "recommended": recommended,
                    "options": ["auto", "search", "retrieval", "sql", "calculation"],
                },
                "execution_trace": [
                    {
                        "node": "supervisor",
                        "decision": f"触发HITL策略确认 | recommended={recommended}",
                    }
                ],
            }
    intent = routing["intent"]
    next_step = "planner"
    decision = (
        f"Supervisor路由: {intent} -> {next_step} | "
        f"reason={routing['reason']} | confidence={routing['confidence']} | question={question}"
    )

    return {
        "next_step": next_step,
        "active_agent": "supervisor",
        "supervisor_decision": decision,
        "execution_trace": [{"node": "supervisor", "decision": decision}],
        "agent_outputs": [
            {
                "agent": "supervisor",
                "intent": intent,
                "selected_agent": next_step,
                "question": question,
                "reason": routing["reason"],
                "confidence": routing["confidence"],
                "should_try_search": routing["should_try_search"],
                "should_try_retrieval": routing["should_try_retrieval"],
            }
        ],
    }
