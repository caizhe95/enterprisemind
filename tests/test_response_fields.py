from graph.agents.field_utils import extract_fields_by_text
from graph.agents.response import _extract_target_fields, response_agent_node
from graph.state_helpers import build_initial_state


class DummyLLM:
    def invoke(self, _prompt):
        raise AssertionError("comparison questions should not call the LLM for field extraction")


def test_comparison_question_skips_field_extraction():
    question = "星澜手机1代和智核笔记本2代哪个更贵，差多少元？"

    assert extract_fields_by_text(question) == []
    assert _extract_target_fields(question, DummyLLM()) == []


def test_multi_field_question_still_extracts_fields():
    question = "星澜手机1代的品类和价格分别是什么？"

    assert extract_fields_by_text(question) == ["品类", "价格"]


def test_descriptive_multi_field_question_extracts_field_phrases():
    question = "星澜手机1代的保修与退货政策关键点分别是什么？"

    assert extract_fields_by_text(question) == ["保修", "退货政策关键点"]


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

    assert result["final_answer"] == "智核笔记本2代更贵，贵1600元。"
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


def test_response_agent_prefers_structured_recommendation_answer():
    state = build_initial_state(
        "预算4000买哪个笔记本？",
        "sess_test",
        "user_test",
    )
    result = response_agent_node(
        {
            **state,
            "question_type": "recommendation",
            "recommendation_context": {
                "recommendations": [
                    {
                        "name": "轻翼笔记本16代",
                        "price": 3699,
                        "reasons": ["属于笔记本", "价格在预算内", "轻薄表现更匹配"],
                    },
                    {
                        "name": "智核笔记本2代",
                        "price": 4399,
                        "reasons": ["属于笔记本", "价格接近预算", "续航表现更匹配"],
                    },
                ]
            },
            "retrieved_docs": [],
        }
    )

    assert "更推荐轻翼笔记本16代" in result["final_answer"]
    assert "备选可以看智核笔记本2代" in result["final_answer"]
    assert result["agent_outputs"][0]["mode"] == "structured_synthesis"
