from graph.agents.response import response_agent_node
from graph.state_helpers import build_initial_state


def test_response_agent_prefers_structured_comparison_answer():
    state = build_initial_state(
        "星澜手机1代和智核笔记本2代哪个更贵，差多少元？",
        "sess_test",
        "user_test",
    )
    result = response_agent_node(
        {
            **state,
            "question_type": "comparison",
            "comparison_context": {
                "metric": "价格",
                "winner": "智核笔记本2代",
                "values": [
                    {"entity": "星澜手机1代", "value": 5499},
                    {"entity": "智核笔记本2代", "value": 7099},
                ],
            },
            "step_results": [
                {
                    "worker": "calculation_agent",
                    "artifacts": {
                        "tool_results": [{"tool": "calculator", "result": "7099 - 5499 = 1600"}]
                    },
                }
            ],
            "retrieved_docs": [],
        }
    )

    assert result["final_answer"] == "智核笔记本2代更贵，智核笔记本2代价格为7099元，星澜手机1代价格为5499元，贵1600元。"
    assert result["agent_outputs"][0]["mode"] == "structured_synthesis"


def test_response_agent_prefers_structured_field_list_answer():
    state = build_initial_state(
        "星澜手机1代的品类和价格分别是什么？",
        "sess_test",
        "user_test",
    )
    result = response_agent_node(
        {
            **state,
            "question_type": "field_list",
            "step_results": [
                {
                    "worker": "extraction_agent",
                    "artifacts": {
                        "fields": {"品类": "手机", "价格": "5499元"},
                        "metrics": {"价格": {"value": 5499, "unit": "元"}},
                    },
                }
            ],
            "retrieved_docs": [],
        }
    )

    assert result["final_answer"] == "品类为手机，价格为5499元。"
    assert result["agent_outputs"][0]["mode"] == "structured_synthesis"
