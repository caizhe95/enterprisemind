"""计算相关节点"""

import re

from graph.state import AgentState
from tools.calculator import calculator


def calculate_node(state: AgentState) -> dict:
    expr_match = re.search(r"[\d\s.+\-*/()]+", state["question"])
    expression = expr_match.group() if expr_match else state["question"]

    try:
        result = calculator.invoke({"expression": expression})
        return {
            "observation": result,
            "tool_results": [{"tool": "calculator", "result": result}],
            "next_step": "response_agent",
        }
    except Exception as e:
        return {"observation": f"计算错误: {e}", "next_step": "response_agent"}


def calculation_agent_node(state: AgentState) -> dict:
    """计算Agent：负责表达式运算与结果结构化"""
    result = calculate_node(state)
    return {
        **result,
        "active_agent": "calculation_agent",
        "agent_outputs": [
            {
                "agent": "calculation_agent",
                "tool": "calculator",
                "observation": result.get("observation"),
            }
        ],
    }
