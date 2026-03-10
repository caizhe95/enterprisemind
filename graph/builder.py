# builder.py - Supervisor多Agent架构版
import os
import urllib.request
from urllib.parse import urlparse
from importlib.metadata import PackageNotFoundError, version
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from graph.state import AgentState
from graph.agents.supervisor import supervisor_node
from graph.agents.sql import sql_agent_node
from graph.agents.search import search_agent_node
from graph.agents.calculation import calculation_agent_node
from graph.agents.retrieval import retrieval_agent_node
from graph.agents.extraction import extraction_agent_node
from graph.agents.recommendation import recommendation_agent_node
from graph.agents.planner import planner_node, orchestrator_node, judge_node, replanner_node
from graph.agents.response import response_agent_node
from graph.agents.hitl import (
    hitl_low_conf_confirm_node,
    hitl_strategy_confirm_node,
    hitl_worker_confirm_node,
)


from config import config


def _ensure_lang_versions():
    """确保 langchain / langgraph 主版本 >= 1"""

    def _major(v: str) -> int:
        part = v.split(".")[0]
        digits = "".join(ch for ch in part if ch.isdigit())
        return int(digits) if digits else 0

    for pkg in ("langchain", "langgraph"):
        try:
            v = version(pkg)
        except PackageNotFoundError as e:
            raise RuntimeError(f"缺少依赖: {pkg}，请先安装 requirements.txt") from e

        if _major(v) < 1:
            raise RuntimeError(f"{pkg} 当前版本为 {v}，需要 >= 1.0.0")


def _append_no_proxy(host: str):
    if not host:
        return
    for key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(key, "")
        items = [x.strip() for x in current.split(",") if x.strip()]
        if host not in items:
            items.append(host)
            os.environ[key] = ",".join(items)


def _langsmith_precheck(endpoint: str, timeout_sec: float) -> tuple[bool, str]:
    """启动前做一次轻量连通性检查，避免运行中反复报 multipart/proxy 错误。"""
    if not endpoint:
        return False, "LANGSMITH_ENDPOINT 为空"

    try:
        parsed = urlparse(endpoint)
        host = parsed.hostname or ""
        _append_no_proxy(host)  # 优先绕过系统代理，避免代理截断 multipart
        health_url = endpoint.rstrip("/") + "/info"
        req = urllib.request.Request(health_url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            code = getattr(resp, "status", 200)
            if 200 <= int(code) < 500:
                return True, f"HTTP {code}"
        return False, "unexpected response"
    except Exception as e:
        return False, str(e)


def setup_langsmith():
    """配置 LangSmith 追踪"""
    if not config.LANGSMITH_TRACING:
        print("🔍 LangSmith 追踪已禁用")
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = config.LANGSMITH_PROJECT

    if config.LANGSMITH_API_KEY:
        os.environ["LANGCHAIN_API_KEY"] = config.LANGSMITH_API_KEY

    if config.LANGSMITH_ENDPOINT:
        os.environ["LANGCHAIN_ENDPOINT"] = config.LANGSMITH_ENDPOINT

    ok, detail = _langsmith_precheck(
        config.LANGSMITH_ENDPOINT, config.LANGSMITH_PRECHECK_TIMEOUT_SEC
    )
    if ok:
        print(f"🔍 LangSmith 追踪已启用 | 项目: {config.LANGSMITH_PROJECT}")
        return True

    if config.LANGSMITH_FAIL_OPEN:
        # 失败自动降级，避免持续刷错影响主流程
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        print(f"⚠️ LangSmith 预检失败，已自动降级关闭追踪: {detail}")
        return False

    raise RuntimeError(f"LangSmith 预检失败: {detail}")


def build_graph():
    """构建 Supervisor 多Agent + Self-RAG 图"""
    _ensure_lang_versions()
    setup_langsmith()

    checkpointer = InMemorySaver()
    store = InMemoryStore()

    workflow = StateGraph(AgentState)

    # Supervisor 与 Specialist Agents
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("judge", judge_node)
    workflow.add_node("replanner", replanner_node)
    workflow.add_node("recommendation_agent", recommendation_agent_node)
    workflow.add_node("sql_agent", sql_agent_node)
    workflow.add_node("search_agent", search_agent_node)
    workflow.add_node("calculation_agent", calculation_agent_node)
    workflow.add_node("retrieval_agent", retrieval_agent_node)
    workflow.add_node("extraction_agent", extraction_agent_node)
    workflow.add_node("response_agent", response_agent_node)
    workflow.add_node("hitl_strategy_confirm", hitl_strategy_confirm_node)
    workflow.add_node("hitl_low_conf_confirm", hitl_low_conf_confirm_node)
    workflow.add_node("hitl_worker_confirm", hitl_worker_confirm_node)

    # 边定义
    workflow.add_edge(START, "supervisor")

    # Supervisor 路由分支
    workflow.add_conditional_edges(
        "supervisor",
        lambda x: x["next_step"],
        {
            "planner": "planner",
            "hitl_strategy_confirm": "hitl_strategy_confirm",
        },
    )

    workflow.add_conditional_edges(
        "hitl_strategy_confirm",
        lambda x: x["next_step"],
        {
            "planner": "planner",
        },
    )

    workflow.add_edge("planner", "orchestrator")
    workflow.add_conditional_edges(
        "orchestrator",
        lambda x: x["next_step"],
        {
            "sql_agent": "sql_agent",
            "search_agent": "search_agent",
            "calculation_agent": "calculation_agent",
            "retrieval_agent": "retrieval_agent",
            "extraction_agent": "extraction_agent",
            "recommendation_agent": "recommendation_agent",
            "response_agent": "response_agent",
        },
    )

    # Specialist -> Response
    workflow.add_conditional_edges(
        "sql_agent",
        lambda x: x["next_step"],
        {"judge": "judge", "response_agent": "response_agent", "end": END},
    )
    workflow.add_conditional_edges(
        "search_agent",
        lambda x: x["next_step"],
        {"judge": "judge", "response_agent": "response_agent"},
    )
    workflow.add_conditional_edges(
        "calculation_agent",
        lambda x: x["next_step"],
        {"judge": "judge", "response_agent": "response_agent"},
    )
    workflow.add_conditional_edges(
        "recommendation_agent",
        lambda x: x["next_step"],
        {"judge": "judge", "response_agent": "response_agent"},
    )
    workflow.add_conditional_edges(
        "retrieval_agent",
        lambda x: x["next_step"],
        {
            "judge": "judge",
            "response_agent": "response_agent",
            "hitl_worker_confirm": "hitl_worker_confirm",
            "end": END,
        },
    )
    workflow.add_conditional_edges(
        "extraction_agent",
        lambda x: x["next_step"],
        {
            "judge": "judge",
            "response_agent": "response_agent",
        },
    )
    workflow.add_edge("hitl_worker_confirm", "retrieval_agent")
    workflow.add_conditional_edges(
        "judge",
        lambda x: x["next_step"],
        {
            "orchestrator": "orchestrator",
            "replanner": "replanner",
            "response_agent": "response_agent",
        },
    )
    workflow.add_conditional_edges(
        "replanner",
        lambda x: x["next_step"],
        {
            "orchestrator": "orchestrator",
            "response_agent": "response_agent",
        },
    )

    # Response Agent: 可结束，或触发补充检索迭代
    workflow.add_conditional_edges(
        "response_agent",
        lambda x: x["next_step"],
        {
            "retrieval_agent": "retrieval_agent",
            "hitl_low_conf_confirm": "hitl_low_conf_confirm",
            "end": END,
        },
    )
    workflow.add_conditional_edges(
        "hitl_low_conf_confirm",
        lambda x: x["next_step"],
        {
            "search_agent": "search_agent",
            "retrieval_agent": "retrieval_agent",
            "end": END,
        },
    )

    compiled_app = workflow.compile(checkpointer=checkpointer, store=store)

    print("✅ Supervisor 多Agent 图编译完成")
    return compiled_app


# 全局应用实例
app = build_graph()
