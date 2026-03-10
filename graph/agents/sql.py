"""SQL相关节点"""

from graph.state import AgentState
from tools.postgres_sql_tool import sql_query, sql_explain, generate_sql_with_examples
from tools.sql_guard import sql_guard
from cache.cache_manager import cache_manager
from graph.agents.worker_contract import build_worker_output


@cache_manager.cached(cache_type="persistent", ttl=86400 * 7, key_prefix="sql_gen")
def generate_sql_cached(question: str) -> str:
    return generate_sql_with_examples(question)


def sql_generate_node(state: AgentState) -> dict:
    try:
        sql = generate_sql_cached(state.get("worker_input") or state["question"])
        return {"generated_sql": sql, "next_step": "sql_check"}
    except Exception as e:
        return {"final_answer": f"SQL生成失败: {e}", "next_step": "end"}


def sql_safety_check_node(state: AgentState) -> dict:
    sql = state.get("generated_sql", "")
    guard_result = sql_guard.invoke({"sql": sql})
    if guard_result.get("allowed"):
        return {
            "next_step": "sql_execute",
            "observation": guard_result.get("reason", "SQL安全检查通过"),
            "tool_results": [{"tool": "sql_guard", "result": guard_result}],
        }

    return {
        "next_step": "sql_explain_only",
        "observation": guard_result.get("reason", "检测到危险SQL"),
        "tool_results": [{"tool": "sql_guard", "result": guard_result}],
    }


def sql_execute_node(state: AgentState) -> dict:
    try:
        result = sql_query.invoke({"question": state.get("worker_input") or state["question"]})
        return {
            "sql_result": result,
            "tool_results": [{"tool": "sql_query", "result": result}],
            "next_step": "response_agent",
        }
    except Exception as e:
        return {"observation": f"SQL错误: {e}", "next_step": "response_agent"}


def sql_explain_only_node(state: AgentState) -> dict:
    result = sql_explain.invoke({"question": state.get("worker_input") or state["question"]})
    return {
        "final_answer": f"生成的SQL（未执行）：\n```sql\n{result}\n```",
        "next_step": "end",
    }


def sql_agent_node(state: AgentState) -> dict:
    """SQL Agent：端到端处理 SQL 生成、安全检查、执行/解释"""
    generated = sql_generate_node(state)
    if generated.get("next_step") == "end":
        return {
            **generated,
            "active_agent": "sql_agent",
            "agent_outputs": [{"agent": "sql_agent", "status": "sql_generate_failed"}],
        }

    merged_state = {**state, **generated}
    checked = sql_safety_check_node(merged_state)

    if checked.get("next_step") == "sql_explain_only":
        explained = sql_explain_only_node(merged_state)
        return {
            **generated,
            **checked,
            **explained,
            "active_agent": "sql_agent",
            "agent_outputs": [{"agent": "sql_agent", "status": "sql_explain_only"}],
        }

    if checked.get("next_step") == "end":
        return {
            **generated,
            **checked,
            "active_agent": "sql_agent",
            "agent_outputs": [{"agent": "sql_agent", "status": "cancelled"}],
        }

    executed = sql_execute_node({**merged_state, **checked})
    combined_tool_results = [*(checked.get("tool_results") or []), *(executed.get("tool_results") or [])]
    next_step = "judge" if state.get("execution_plan") else "response_agent"
    normalized_output = build_worker_output(
        worker="sql_agent",
        status="success" if executed.get("sql_result") else "partial",
        summary=executed.get("observation", "SQL执行完成"),
        artifacts={
            "generated_sql": generated.get("generated_sql"),
            "sql_result": executed.get("sql_result"),
            "tool_results": combined_tool_results,
        },
        signals=["analytics_ready"] if executed.get("sql_result") else [],
        confidence=0.85 if executed.get("sql_result") else 0.3,
    )
    return {
        **generated,
        **checked,
        **executed,
        "tool_results": combined_tool_results,
        "next_step": next_step,
        "active_agent": "sql_agent",
        "last_worker_output": normalized_output,
        "agent_outputs": [
            {"agent": "sql_agent", "has_sql_result": bool(executed.get("sql_result"))}
        ],
    }
