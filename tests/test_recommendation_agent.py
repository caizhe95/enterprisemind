from graph.agents.recommendation import recommendation_agent_node
from graph.agents.extraction import extraction_agent_node
from graph.agents.retrieval import _prioritize_recommendation_docs
from graph.agents.search import _normalize_tavily_docs
from graph.agents.sql import sql_safety_check_node
from graph.state_helpers import build_initial_state


def test_recommendation_agent_ranks_budget_candidates():
    state = build_initial_state("预算4000买哪个笔记本？", "sess_test", "user_test")
    products = [
        {"name": "轻翼笔记本16代", "category": "笔记本", "price": 3699, "highlights": ["轻薄便携", "长续航", "屏幕素质高"]},
        {"name": "智核笔记本2代", "category": "笔记本", "price": 4399, "highlights": ["长续航", "屏幕素质高", "轻薄便携"]},
    ]

    result = recommendation_agent_node({**state, "extraction_context": {"products": products}, "execution_plan": [{}]})

    recommendations = result["recommendation_context"]["recommendations"]
    assert recommendations[0]["name"] == "轻翼笔记本16代"
    assert result["last_worker_output"]["signals"] == ["recommendation_ready"]
    assert any(item["tool"] == "candidate_ranker" for item in result["tool_results"])


def test_retrieval_prioritizes_shopping_guides_for_recommendation_queries():
    docs = [
        {"content": "普通产品片段", "metadata": {"file_name": "products.md", "rrf_score": 0.2}},
        {
            "content": "推荐商品: 智核笔记本2代\n推荐理由: 长续航、轻薄便携",
            "metadata": {"file_name": "guides/shopping_guide_0002.md", "rrf_score": 0.1},
        },
    ]

    ranked = _prioritize_recommendation_docs("预算4000买哪个笔记本？", docs)
    assert ranked[0]["metadata"]["file_name"] == "guides/shopping_guide_0002.md"


def test_recommendation_agent_applies_game_scenario_weights():
    state = build_initial_state("预算9000买哪个游戏笔记本？", "sess_test", "user_test")
    products = [
        {"name": "轻翼笔记本16代", "category": "笔记本", "price": 7999, "highlights": ["轻薄便携", "长续航", "屏幕素质高"]},
        {"name": "飞拓笔记本18代", "category": "笔记本", "price": 8299, "highlights": ["高性能", "长续航", "散热稳定"]},
    ]

    result = recommendation_agent_node({**state, "extraction_context": {"products": products}, "execution_plan": [{}]})
    recommendations = result["recommendation_context"]["recommendations"]
    assert recommendations[0]["name"] == "飞拓笔记本18代"


def test_recommendation_agent_applies_office_scenario_weights():
    state = build_initial_state("预算5000买哪个办公笔记本？", "sess_test", "user_test")
    products = [
        {"name": "轻翼笔记本16代", "category": "笔记本", "price": 4699, "highlights": ["轻薄便携", "长续航", "屏幕素质高"]},
        {"name": "飞拓笔记本18代", "category": "笔记本", "price": 4899, "highlights": ["高性能", "散热稳定", "长续航"]},
    ]

    result = recommendation_agent_node({**state, "extraction_context": {"products": products}, "execution_plan": [{}]})
    recommendations = result["recommendation_context"]["recommendations"]
    assert recommendations[0]["name"] == "轻翼笔记本16代"


def test_extraction_agent_normalizes_numeric_price_field():
    state = build_initial_state("星澜手机1代的售价和品类分别是什么？", "sess_test", "user_test")
    docs = [
        {
            "content": "\n".join(
                [
                    "## 星澜手机1代",
                    "- 品类: 手机",
                    "售价: 5499",
                ]
            ),
            "metadata": {},
        }
    ]

    result = extraction_agent_node({**state, "retrieved_docs": docs, "execution_plan": [{}]})

    assert result["extraction_context"]["fields"]["价格"] == "5499元"
    assert any(item["tool"] == "field_normalizer" for item in result["tool_results"])


def test_sql_guard_blocks_dangerous_sql():
    result = sql_safety_check_node({"generated_sql": "DELETE FROM sales;", "question": "", "worker_input": None})

    assert result["next_step"] == "sql_explain_only"
    assert result["tool_results"][0]["tool"] == "sql_guard"


def test_search_normalizes_dict_style_tavily_response():
    result = {
        "answer": "推荐先看轻薄本",
        "results": [
            {
                "title": "导购文章",
                "content": "轻薄本更适合通勤和办公场景",
                "url": "https://example.com/guide",
            }
        ],
    }

    docs = _normalize_tavily_docs(result)

    assert docs[0]["metadata"]["source"] == "tavily_answer"
    assert docs[1]["metadata"]["source"] == "https://example.com/guide"
