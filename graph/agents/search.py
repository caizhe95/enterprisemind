"""搜索相关节点"""

from __future__ import annotations

import asyncio
from typing import Any

from graph.state import AgentState
from tools.tavily_tool import tavily_search, tavily_search_async

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


def _dedupe_docs(docs: list[dict]) -> list[dict]:
    merged = []
    seen = set()
    for doc in docs:
        key = (
            doc.get("metadata", {}).get("source"),
            doc.get("metadata", {}).get("title"),
            doc.get("content"),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)
    return merged


async def _search_node_async(state: AgentState) -> dict:
    try:
        query = state.get("worker_input") or state["question"]
        basic_result, advanced_result = await asyncio.gather(
            tavily_search_async(query=query, search_depth="basic", max_results=5),
            tavily_search_async(query=query, search_depth="advanced", max_results=5),
        )

        docs = _dedupe_docs(
            _normalize_tavily_docs(basic_result) + _normalize_tavily_docs(advanced_result)
        )
        eval_result = None
        if docs:
            eval_result = get_self_rag_evaluator().evaluate_retrieval(
                state["question"], docs
            )

        if eval_result is None:
            eval_result = {"details": docs}

        return {
            "retrieved_docs": eval_result["details"],
            "tool_results": [
                {
                    "tool": "tavily_search",
                    "result": {
                        "basic": basic_result,
                        "advanced": advanced_result,
                    },
                }
            ],
            "next_step": "response_agent",
        }
    except Exception:
        # Fallback to sync tool to preserve compatibility in restricted envs.
        try:
            result = tavily_search.invoke(
                {
                    "query": state.get("worker_input") or state["question"],
                    "search_depth": "basic",
                    "max_results": 5,
                }
            )
            docs = _normalize_tavily_docs(result)
            return {
                "retrieved_docs": docs,
                "tool_results": [{"tool": "tavily_search", "result": result}],
                "next_step": "response_agent",
            }
        except Exception as exc:
            return {"observation": f"搜索失败: {exc}", "next_step": "response_agent"}


def search_node(state: AgentState) -> dict:
    return asyncio.run(_search_node_async(state))


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
