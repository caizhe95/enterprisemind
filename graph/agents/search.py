"""搜索相关节点"""

from __future__ import annotations

from typing import Any

from graph.state import AgentState
from tools.tavily_tool import tavily_search

from graph.agents.common import get_self_rag_evaluator
from graph.agents.worker_contract import build_worker_output


def _normalize_tavily_docs(result: Any) -> list[dict]:
    docs = []

    if isinstance(result, dict):
        answer = str(result.get("answer") or "").strip()
        if answer:
            docs.append({"content": answer, "metadata": {"source": "tavily_answer"}})
        for item in result.get("results") or []:
            content = str(item.get("content") or item.get("raw_content") or "").strip()
            if not content:
                continue
            docs.append(
                {
                    "content": content[:500],
                    "metadata": {
                        "source": item.get("url") or "tavily",
                        "title": item.get("title"),
                    },
                }
            )
        return docs

    text = str(result or "")
    if "来源:" in text:
        for line in text.split("来源:")[1:]:
            cleaned = line.strip()
            if cleaned:
                docs.append({"content": cleaned[:500], "metadata": {"source": "tavily"}})
        return docs

    if text.strip():
        docs.append({"content": text[:500], "metadata": {"source": "tavily"}})
    return docs


def search_node(state: AgentState) -> dict:
    try:
        result = tavily_search.invoke(
            {
                "query": state.get("worker_input") or state["question"],
                "search_depth": "basic",
                "max_results": 5,
            }
        )

        docs = _normalize_tavily_docs(result)
        eval_result = None

        if docs:
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
    next_step = "judge" if state.get("execution_plan") else result.get("next_step", "response_agent")
    normalized_output = build_worker_output(
        worker="search_agent",
        status="success" if result.get("retrieved_docs") else "partial",
        summary=result.get("observation", "搜索完成"),
        artifacts={
            "retrieved_docs": result.get("retrieved_docs", []),
            "tool_results": result.get("tool_results", []),
        },
        signals=["documents_found"] if result.get("retrieved_docs") else [],
        confidence=0.8 if result.get("retrieved_docs") else 0.3,
    )
    return {
        **result,
        "next_step": next_step,
        "active_agent": "search_agent",
        "last_worker_output": normalized_output,
        "agent_outputs": [
            {
                "agent": "search_agent",
                "tool": "tavily_search",
                "retrieved_docs": len(result.get("retrieved_docs", [])),
            }
        ],
    }
