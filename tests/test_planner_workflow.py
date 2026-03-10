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
