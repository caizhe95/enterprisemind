"""搜索相关节点"""

from graph.state import AgentState
from tools.tavily_tool import tavily_search

from graph.agents.common import get_self_rag_evaluator


def search_node(state: AgentState) -> dict:
    try:
        result = tavily_search.invoke(
            {"query": state["question"], "search_depth": "basic", "max_results": 5}
        )

        docs = []
        eval_result = None

        if "来源:" in result:
            for line in result.split("来源:")[1:]:
                docs.append({"content": line[:500], "metadata": {"source": "tavily"}})
            eval_result = get_self_rag_evaluator().evaluate_retrieval(
                state["question"], docs
            )

        if eval_result is None:
            eval_result = {"details": docs}

        return {
            "retrieved_docs": eval_result["details"],
            "tool_results": [{"tool": "tavily_search", "result": result}],
            "next_step": "response_agent",
        }
    except Exception as e:
        return {"observation": f"搜索失败: {e}", "next_step": "response_agent"}


def search_agent_node(state: AgentState) -> dict:
    """搜索Agent：负责互联网实时搜索"""
    result = search_node(state)
    return {
        **result,
        "active_agent": "search_agent",
        "agent_outputs": [
            {
                "agent": "search_agent",
                "tool": "tavily_search",
                "retrieved_docs": len(result.get("retrieved_docs", [])),
            }
        ],
    }
