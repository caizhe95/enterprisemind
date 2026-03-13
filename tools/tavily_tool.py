"""Tavily搜索。"""

from __future__ import annotations

import os
from typing import Annotated, Any

import httpx
from langchain_core.tools import tool
from tavily import TavilyClient

_tavily_client = None
TAVILY_SEARCH_URL = "https://api.tavily.com/search"


def get_client():
    global _tavily_client
    if _tavily_client is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY not set")
        _tavily_client = TavilyClient(api_key=api_key)
    return _tavily_client


async def tavily_search_async(
    query: str,
    search_depth: str = "basic",
    max_results: int = 5,
) -> dict[str, Any]:
    """Async Tavily search using direct HTTP call for non-blocking I/O."""

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not set")

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": search_depth,
        "max_results": max_results,
        "include_answer": True,
        "include_raw_content": True,
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(TAVILY_SEARCH_URL, json=payload)
        response.raise_for_status()
        return response.json()


@tool
def tavily_search(
    query: Annotated[str, "搜索查询词"],
    search_depth: Annotated[str, "'basic'或'advanced'"] = "basic",
    max_results: Annotated[int, "结果数量"] = 5,
) -> str:
    """互联网实时搜索，获取最新信息"""

    client = get_client()

    response = client.search(
        query=query,
        search_depth=search_depth,
        max_results=max_results,
        include_answer=True,
        include_raw_content=True,
    )

    lines = [f"搜索: {query}", f"答案: {response.get('answer', '')}", "\n来源:"]

    for i, src in enumerate(response.get("results", [])[:3], 1):
        lines.append(f"{i}. [{src.get('title')}] {src.get('content')[:300]}...")

    return "\n".join(lines)
