"""SQL相关节点"""

from graph.state import AgentState
from tools.postgres_sql_tool import sql_query, sql_explain, generate_sql_with_examples
from cache.cache_manager import cache_manager


@cache_manager.cached(cache_type="persistent", ttl=86400 * 7, key_prefix="sql_gen")
def generate_sql_cached(question: str) -> str:
    return generate_sql_with_examples(question)


def sql_generate_node(state: AgentState) -> dict:
    try:
        sql = generate_sql_cached(state["question"])
        return {"generated_sql": sql, "next_step": "sql_check"}
    except Exception as e:
        return {"final_answer": f"SQL生成失败: {e}", "next_step": "end"}


def sql_safety_check_node(state: AgentState) -> dict:
    sql = state.get("generated_sql", "")
    dangerous = ["DELETE", "UPDATE", "INSERT", "DROP", "ALTER", "TRUNCATE"]
    is_dangerous = any(d in sql.upper() for d in dangerous)

    if not is_dangerous:
        return {"next_step": "sql_execute", "observation": "SQL安全检查通过"}

    return {
        "next_step": "sql_explain_only",
        "observation": "检测到危险SQL，用户侧仅允许查询，已自动降级为仅解释SQL",
    }


def sql_execute_node(state: AgentState) -> dict:
    try:
        result = sql_query.invoke({"question": state["question"]})
        return {
            "sql_result": result,
            "tool_results": [{"tool": "sql_query", "result": result}],
            "next_step": "response_agent",
        }
    except Exception as e:
        return {"observation": f"SQL错误: {e}", "next_step": "response_agent"}


def sql_explain_only_node(state: AgentState) -> dict:
    result = sql_explain.invoke({"question": state["question"]})
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
    return {
        **generated,
        **checked,
        **executed,
        "next_step": "response_agent",
        "active_agent": "sql_agent",
        "agent_outputs": [
            {"agent": "sql_agent", "has_sql_result": bool(executed.get("sql_result"))}
        ],
    }
