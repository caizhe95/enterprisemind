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
            "artifacts": {"metrics": {"价格": {"value": 5499, "unit": "元"}}},
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
            "artifacts": {"metrics": {"价格": {"value": 7099, "unit": "元"}}},
            "signals": ["documents_found", "value_found", "structured_data_ready"],
            "confidence": 0.9,
            "errors": [],
        },
        "step_results": (after_first.get("step_results") or []) + judged_first["step_results"],
    }
    judged_second = judge_node(after_second)

    assert judged_second["calculation_expression"] == "7099 - 5499"
    assert judged_second["comparison_context"]["winner"] == "智核笔记本2代"


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
