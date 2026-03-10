"""计算相关节点"""

import re

from graph.state import AgentState
from tools.calculator import calculator
from graph.agents.worker_contract import build_worker_output


def calculate_node(state: AgentState) -> dict:
    expression = state.get("calculation_expression") or state.get("worker_input")
    if not expression:
        expr_match = re.search(r"[\d\s.+\-*/()]+", state["question"])
        expression = expr_match.group() if expr_match else state["question"]

    try:
        result = calculator.invoke({"expression": expression})
        return {
            "observation": result,
            "tool_results": [{"tool": "calculator", "result": result, "expression": expression}],
            "next_step": "response_agent",
        }
    except Exception as e:
        return {"observation": f"计算错误: {e}", "next_step": "response_agent"}


def calculation_agent_node(state: AgentState) -> dict:
    """计算Agent：负责表达式运算与结果结构化"""
    result = calculate_node(state)
    next_step = "judge" if state.get("execution_plan") else result.get("next_step", "response_agent")
    normalized_output = build_worker_output(
        worker="calculation_agent",
        status="success" if "错误" not in str(result.get("observation", "")) else "failed",
        summary=str(result.get("observation", "计算完成")),
        artifacts={
            "expression": state.get("calculation_expression") or state.get("worker_input"),
            "tool_results": result.get("tool_results", []),
            "comparison_context": state.get("comparison_context"),
        },
        signals=["calculation_done"] if "错误" not in str(result.get("observation", "")) else [],
        confidence=0.9 if "错误" not in str(result.get("observation", "")) else 0.0,
        errors=[] if "错误" not in str(result.get("observation", "")) else [str(result.get("observation"))],
    )
    return {
        **result,
        "next_step": next_step,
        "active_agent": "calculation_agent",
        "last_worker_output": normalized_output,
        "agent_outputs": [
            {
                "agent": "calculation_agent",
                "tool": "calculator",
                "observation": result.get("observation"),
            }
        ],
    }
