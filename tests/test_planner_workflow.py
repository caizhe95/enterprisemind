from graph.agents.common import analyze_intent
from graph.agents import common as common_module
from graph.agents.supervisor import supervisor_node
from graph.agents.planner import judge_node, planner_node, replanner_node
from graph.state_helpers import build_initial_state


def test_planner_builds_multi_agent_plan_for_comparison():
    state = build_initial_state(
        "星澜手机1代和智核笔记本2代哪个更贵，差多少元？",
        "sess_test",
        "user_test",
    )

    result = planner_node(state)
    plan = result["execution_plan"]

    assert result["question_type"] == "comparison"
    assert [step["worker"] for step in plan] == [
        "retrieval_agent",
        "extraction_agent",
        "retrieval_agent",
        "extraction_agent",
        "calculation_agent",
        "response_agent",
    ]
    assert plan[0]["input"] == "星澜手机1代 价格"
    assert plan[2]["input"] == "智核笔记本2代 价格"


def test_planner_builds_single_agent_plan_for_single_fact():
    state = build_initial_state(
        "智核笔记本2代属于什么品类？",
        "sess_test",
        "user_test",
    )

    result = planner_node(state)
    assert result["question_type"] == "single_fact"
    assert [step["worker"] for step in result["execution_plan"]] == [
        "retrieval_agent",
        "extraction_agent",
        "response_agent",
    ]


def test_planner_builds_recommendation_plan():
    state = build_initial_state(
        "预算4000买哪个笔记本？",
        "sess_test",
        "user_test",
    )

    result = planner_node(state)
    assert result["question_type"] == "recommendation"
    assert [step["worker"] for step in result["execution_plan"]] == [
        "retrieval_agent",
        "extraction_agent",
        "recommendation_agent",
        "response_agent",
    ]


def test_judge_prepares_calculation_expression_after_two_extraction_steps():
    state = build_initial_state(
        "星澜手机1代和智核笔记本2代哪个更贵，差多少元？",
        "sess_test",
        "user_test",
    )
    planned = planner_node(state)
    plan = planned["execution_plan"]

    after_first = {
        **state,
        **planned,
        "current_step_index": 1,
        "worker_input": plan[1]["input"],
        "last_worker_output": {
            "worker": "extraction_agent",
            "status": "success",
            "summary": "结构化抽取完成",
            "artifacts": {
                "metrics": {"价格": {"value": 5499, "unit": "元"}},
            },
            "signals": ["documents_found", "value_found", "structured_data_ready"],
            "confidence": 0.9,
            "errors": [],
        },
        "step_results": [],
    }
    judged_first = judge_node(after_first)

    after_second = {
        **after_first,
        **judged_first,
        "current_step_index": 3,
        "worker_input": plan[3]["input"],
        "last_worker_output": {
            "worker": "extraction_agent",
            "status": "success",
            "summary": "结构化抽取完成",
            "artifacts": {
                "metrics": {"价格": {"value": 7099, "unit": "元"}},
            },
            "signals": ["documents_found", "value_found", "structured_data_ready"],
            "confidence": 0.9,
            "errors": [],
        },
        "step_results": (after_first.get("step_results") or []) + judged_first["step_results"],
    }
    judged_second = judge_node(after_second)

    assert judged_second["calculation_expression"] == "7099 - 5499"
    assert judged_second["comparison_context"]["winner"] == "智核笔记本2代"


def test_judge_stops_workflow_when_worker_contract_reports_failure():
    state = build_initial_state("星澜手机1代和智核笔记本2代哪个更贵？", "sess_test", "user_test")
    planned = planner_node(state)

    judged = judge_node(
        {
            **state,
            **planned,
            "current_step_index": 0,
            "worker_input": planned["execution_plan"][0]["input"],
            "last_worker_output": {
                "worker": "retrieval_agent",
                "status": "failed",
                "summary": "检索失败",
                "artifacts": {},
                "signals": [],
                "confidence": 0.0,
                "errors": ["network error"],
            },
            "step_results": [],
        }
    )

    assert judged["next_step"] == "replanner"


def test_replanner_retries_retrieval_with_expanded_query_first():
    state = build_initial_state("星澜手机1代和智核笔记本2代哪个更贵？", "sess_test", "user_test")
    planned = planner_node(state)

    replanned = replanner_node(
        {
            **state,
            **planned,
            "current_step_index": 0,
            "replan_reason": "worker_status=failed, unmet_expects=value_found",
            "step_retry_counts": {},
        }
    )

    step = replanned["execution_plan"][0]
    assert replanned["next_step"] == "orchestrator"
    assert replanned["step_retry_counts"]["0"] == 1
    assert step["worker"] == "retrieval_agent"
    assert "星澜手机1代 价格" in step["input"]


def test_replanner_switches_retrieval_to_search_after_retry_exhausted():
    state = build_initial_state("星澜手机1代和智核笔记本2代哪个更贵？", "sess_test", "user_test")
    planned = planner_node(state)

    replanned = replanner_node(
        {
            **state,
            **planned,
            "current_step_index": 0,
            "replan_reason": "worker_status=partial, unmet_expects=value_found",
            "step_retry_counts": {"0": 1},
        }
    )

    step = replanned["execution_plan"][0]
    assert step["worker"] == "search_agent"
    assert replanned["step_retry_counts"]["0"] == 2


def test_planner_resets_stale_execution_artifacts():
    state = build_initial_state("预算4000买哪个笔记本？", "sess_test", "user_test")
    dirty_state = {
        **state,
        "retrieved_docs": [{"content": "old", "metadata": {}}],
        "tool_results": [{"tool": "old_tool", "result": "stale"}],
        "step_results": [{"worker": "old_worker"}],
        "comparison_context": {"winner": "旧商品"},
        "recommendation_context": {"recommendations": [{"name": "旧推荐"}]},
        "extraction_context": {"fields": {"价格": "999元"}},
    }

    result = planner_node(dirty_state)

    assert result["retrieved_docs"] == []
    assert result["tool_results"] == []
    assert result["step_results"] == []
    assert result["comparison_context"] is None
    assert result["recommendation_context"] is None
    assert result["extraction_context"] is None


def test_replanner_clears_tool_artifacts_before_retry():
    state = build_initial_state("星澜手机1代和智核笔记本2代哪个更贵？", "sess_test", "user_test")
    planned = planner_node(state)

    replanned = replanner_node(
        {
            **state,
            **planned,
            "current_step_index": 0,
            "replan_reason": "worker_status=failed, unmet_expects=value_found",
            "step_retry_counts": {},
            "tool_results": [{"tool": "old_tool", "result": "stale"}],
            "tool_calls": [{"tool": "old_tool"}],
        }
    )

    assert replanned["tool_results"] == []
    assert replanned["tool_calls"] == []


def test_open_domain_product_fact_query_allows_search_fallback():
    analysis = analyze_intent("iPhone 15 的起售价是多少？")

    assert analysis["intent"] == "retrieval"
    assert analysis["should_try_search"] is True
    assert analysis["auto_route_to_search_on_dual"] is True


def test_internal_catalog_product_fact_stays_on_local_retrieval():
    analysis = analyze_intent("星澜手机1代的价格")

    assert analysis["intent"] == "retrieval"
    assert analysis["should_try_search"] is False
    assert analysis["auto_route_to_search_on_dual"] is False


def test_supervisor_auto_routes_open_domain_product_fact_to_search_without_hitl():
    state = build_initial_state("iPhone 15 的起售价是多少？", "sess_test", "user_test")

    routed = supervisor_node(state)

    assert routed["next_step"] == "planner"
    assert routed["agent_outputs"][0]["intent"] == "search"


def test_judge_sends_low_relevance_open_domain_retrieval_to_replanner():
    state = build_initial_state("iPhone 15 的起售价是多少？", "sess_test", "user_test")
    planned = planner_node(state)

    judged = judge_node(
        {
            **state,
            **planned,
            "current_step_index": 0,
            "worker_input": planned["execution_plan"][0]["input"],
            "last_worker_output": {
                "worker": "retrieval_agent",
                "status": "success",
                "summary": "检索完成",
                "artifacts": {
                    "retrieved_docs": [{"content": "云途手机15代 价格: 2999元", "metadata": {}}],
                    "retrieval_grade": "partially_relevant",
                },
                "signals": ["documents_found"],
                "confidence": 0.6,
                "errors": [],
            },
            "step_results": [],
        }
    )

    assert judged["next_step"] == "replanner"
    assert judged["replan_reason"] == "fallback_to_search_due_to_low_relevance_or_empty_extraction"


def test_replanner_switches_extraction_step_to_search_for_open_domain_fact():
    state = build_initial_state("iPhone 15 的起售价是多少？", "sess_test", "user_test")
    planned = planner_node(state)

    replanned = replanner_node(
        {
            **state,
            **planned,
            "current_step_index": 1,
            "replan_reason": "fallback_to_search_due_to_low_relevance_or_empty_extraction",
            "step_retry_counts": {},
        }
    )

    assert replanned["execution_plan"][1]["worker"] == "search_agent"
    assert replanned["execution_plan"][2]["worker"] == "extraction_agent"


def test_low_confidence_rule_route_can_be_refined_by_llm(monkeypatch):
    monkeypatch.setattr(common_module.config, "ENABLE_LLM_INTENT_ROUTING", True)
    monkeypatch.setattr(
        common_module,
        "_analyze_intent_with_llm",
        lambda question, rule_analysis: {
            "intent": "search",
            "reason": "LLM判断为公开概念问答",
            "confidence": "high",
            "should_try_search": True,
            "should_try_retrieval": False,
            "auto_route_to_search_on_dual": False,
            "route_source": "llm",
        },
    )

    analysis = analyze_intent("MCP 协议怎么理解")

    assert analysis["intent"] == "search"
    assert analysis["route_source"] == "llm"


def test_high_confidence_internal_attribute_query_stays_rule_routed(monkeypatch):
    monkeypatch.setattr(common_module.config, "ENABLE_LLM_INTENT_ROUTING", True)
    monkeypatch.setattr(
        common_module,
        "_analyze_intent_with_llm",
        lambda question, rule_analysis: {
            "intent": "search",
            "reason": "误判到公网",
            "confidence": "high",
            "should_try_search": True,
            "should_try_retrieval": False,
            "auto_route_to_search_on_dual": False,
            "route_source": "llm",
        },
    )

    analysis = analyze_intent("星澜手机1代的价格是多少")

    assert analysis["intent"] == "retrieval"
    assert analysis["route_source"] == "rules"
