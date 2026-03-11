from graph.agents.recommendation import recommendation_agent_node
from graph.agents.extraction import extraction_agent_node
from graph.agents.retrieval import (
    _prioritize_entity_precise_docs,
    _prioritize_recommendation_docs,
    _react_retrieve,
    _prioritize_section_docs,
)
from graph.agents.section_utils import get_section_synonym_groups
from graph.agents.field_utils import canonicalize_field_name
from graph.agents.search import _normalize_tavily_docs
from graph.agents.sql import sql_safety_check_node
from rag.retrieval_engine import RetrievalEngine
from graph.state_helpers import build_initial_state
from tools.knowledge_data_validator import validate_product_records
from tools.structured_extractor import structured_extractor
from tools.rerank_tool import rerank_tool


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


def test_section_synonyms_are_loaded_from_config():
    groups = get_section_synonym_groups()

    assert ("退换货", "退货", "换货") in groups


def test_field_aliases_are_loaded_from_config():
    assert canonicalize_field_name("售价") == "价格"
    assert canonicalize_field_name("上市时间") == "发布时间"


def test_rerank_tool_does_not_boost_guides_for_single_fact_queries():
    docs = [
        {
            "content": "## 星澜手机1代\n- 品类: 手机\n- 价格: 5499元",
            "metadata": {"file_name": "products.md", "rrf_score": 0.2},
        },
        {
            "content": "推荐商品: 星澜手机1代\n推荐理由: 预算区间3000-6000元",
            "metadata": {"file_name": "guides/shopping_guide_0001.md", "rrf_score": 0.2},
        },
    ]

    ranked = rerank_tool.invoke({"query": "星澜手机1代的价格", "docs": docs, "top_k": 2})["docs"]

    assert ranked[0]["metadata"]["file_name"] == "products.md"


def test_retrieval_prioritizes_exact_entity_over_wrong_generation_docs():
    state = build_initial_state("智核笔记本2代的价格是多少？", "sess_test", "user_test")
    state["question_type"] = "single_fact"
    docs = [
        {
            "content": "## 智核笔记本8代\n- 价格: 8999元",
            "metadata": {"file_name": "products.md", "rrf_score": 0.9},
        },
        {
            "content": "## 智核笔记本2代\n- 价格: 4399元",
            "metadata": {"file_name": "products.md", "rrf_score": 0.5},
        },
    ]

    ranked = _prioritize_entity_precise_docs(state, state["question"], docs)

    assert ranked[0]["content"].startswith("## 智核笔记本2代")


def test_retrieval_prefers_product_card_over_sales_for_single_fact():
    state = build_initial_state("智核笔记本2代的价格是多少？", "sess_test", "user_test")
    state["question_type"] = "single_fact"
    docs = [
        {
            "content": "记录1: 日期=2024-01-02 ; 产品名称=智核笔记本2代 ; 销售额=56792",
            "metadata": {"file_name": "sales.md", "rrf_score": 0.9},
        },
        {
            "content": "## 智核笔记本2代\n- 品类: 笔记本\n- 价格: 4399元",
            "metadata": {"file_name": "products.md", "rrf_score": 0.3},
        },
    ]

    ranked = _prioritize_entity_precise_docs(state, state["question"], docs)

    assert ranked[0]["metadata"]["file_name"] == "products.md"


def test_react_retrieve_keeps_single_fact_followup_query_plain(monkeypatch):
    captured_queries = []

    class FakeEngine:
        def hybrid_search(self, query, top_k=8):
            captured_queries.append(query)
            return []

    monkeypatch.setattr("rag.retrieval_engine.RetrievalEngine", lambda: FakeEngine())
    monkeypatch.setattr("graph.agents.retrieval.config.REACT_MAX_STEPS", 2)

    state = build_initial_state("星澜手机1代的价格是多少？", "sess_test", "user_test")
    state["question_type"] = "single_fact"

    _react_retrieve(state)

    assert captured_queries == ["星澜手机1代的价格是多少？", "星澜手机1代的价格是多少？"]


def test_prioritize_section_docs_boosts_section_matched_docs_for_cross_doc_query():
    docs = [
        {
            "content": "推荐商品: 星澜手机1代",
            "metadata": {"file_name": "shopping_guide_0001.md", "rrf_score": 0.8},
        },
        {
            "content": "1. 签收后7天内，商品完好且配件齐全支持无理由退货。",
            "metadata": {"file_name": "policies.md", "header_path": "公司政策 > 退换货政策", "rrf_score": 0.1},
        },
        {
            "content": "- 保修: 整机1年",
            "metadata": {"file_name": "products.md", "header_path": "产品信息 > 星澜手机1代", "rrf_score": 0.2},
        },
    ]

    ranked = _prioritize_section_docs("星澜手机1代的保修与退货政策关键点分别是什么", docs)

    assert ranked[0]["metadata"]["file_name"] in {"policies.md", "products.md"}
    assert ranked[1]["metadata"]["file_name"] in {"policies.md", "products.md"}


def test_retrieval_engine_exact_entity_recall_prefers_product_docs():
    engine = RetrievalEngine.__new__(RetrievalEngine)
    engine.documents_cache = [
        {
            "content": "记录1: 日期=2024-01-02 ; 产品名称=智核笔记本2代 ; 数量=8 ; 销售额=35192",
            "metadata": {"file_name": "sales.md", "chunk_id": "sales_1"},
        },
        {
            "content": "## 智核笔记本2代\n- 品类: 笔记本\n- 价格: 4399元",
            "metadata": {"file_name": "products.md", "chunk_id": "prod_1"},
        },
        {
            "content": "## 智核笔记本8代\n- 品类: 笔记本\n- 价格: 8999元",
            "metadata": {"file_name": "products.md", "chunk_id": "prod_2"},
        },
    ]

    results = RetrievalEngine._exact_entity_recall(engine, "智核笔记本2代的价格是多少？", top_k=5)

    assert results[0]["metadata"]["file_name"] == "products.md"
    assert "智核笔记本2代" in results[0]["content"]


def test_extract_query_entity_strips_temporal_modifier():
    from graph.agents.retrieval import _extract_query_entity as retrieval_entity
    from tools.structured_extractor import _extract_query_entity as extractor_entity

    query = "智核笔记本2代现在价格是多少？"
    assert retrieval_entity(query) == "智核笔记本2代"
    assert extractor_entity(query) == "智核笔记本2代"


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


def test_recommendation_agent_prefers_strict_budget_pool_when_available():
    state = build_initial_state("预算5000买哪个笔记本？", "sess_test", "user_test")
    products = [
        {"name": "轻翼笔记本16代", "category": "笔记本", "price": 4699, "highlights": ["轻薄便携", "长续航"]},
        {"name": "飞拓笔记本12代", "category": "笔记本", "price": 5899, "highlights": ["轻薄便携", "长续航"]},
    ]

    result = recommendation_agent_node({**state, "extraction_context": {"products": products}, "execution_plan": [{}]})

    recommendations = result["recommendation_context"]["recommendations"]
    assert recommendations[0]["name"] == "轻翼笔记本16代"
    assert result["recommendation_context"]["selection_summary"]["has_hard_budget_match"] is True
    assert result["recommendation_context"]["excluded_candidates"][0]["reasons"] == ["over_budget"]


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


def test_extraction_agent_infers_metric_for_single_fact_price_query():
    state = build_initial_state("星澜手机1代的价格是多少？", "sess_test", "user_test")
    docs = [
        {
            "content": "## 星澜手机1代\n- 品类: 手机\n- 价格: 5499元",
            "metadata": {"file_name": "products.md"},
        }
    ]

    result = extraction_agent_node({**state, "retrieved_docs": docs, "execution_plan": [{}]})

    assert result["extraction_context"]["metrics"]["价格"]["value"] == 5499
    assert "value_found" in result["last_worker_output"]["signals"]


def test_structured_extractor_prefers_exact_entity_in_docs():
    docs = [
        {"content": "## 智核笔记本8代\n- 价格: 8999元", "metadata": {}},
        {"content": "## 智核笔记本2代\n- 价格: 4399元", "metadata": {}},
    ]

    result = structured_extractor.invoke({"query": "智核笔记本2代的价格是多少？", "docs": docs, "metric": "价格"})

    assert result["metrics"]["价格"]["value"] == 4399


def test_structured_extractor_derives_price_from_sales_records():
    docs = [
        {
            "content": "记录1: 日期=2024-01-02 ; 产品名称=智核笔记本2代 ; 品类=笔记本 ; 数量=8 ; 销售额=35192",
            "metadata": {"file_name": "sales.md"},
        }
    ]

    result = structured_extractor.invoke({"query": "智核笔记本2代的价格是多少？", "docs": docs, "metric": "价格"})

    assert result["metrics"]["价格"]["value"] == 4399


def test_structured_extractor_reads_policy_section_from_metadata_header():
    docs = [
        {
            "content": "1. 签收后7天内，商品完好且配件齐全支持无理由退货。\n2. 签收后15天内出现质量问题支持换货。",
            "metadata": {"file_name": "policies.md", "H2": "退换货政策", "header_path": "公司政策 > 退换货政策"},
        }
    ]

    result = structured_extractor.invoke(
        {"query": "星澜手机1代的退货政策关键点是什么", "docs": docs, "metric": None}
    )

    assert result["fields"]["退货政策关键点"] == "签收后7天内，商品完好且配件齐全支持无理由退货。"


def test_knowledge_data_validator_reports_duplicate_product_names():
    records = [
        {"name": "星澜手机1代", "SKU": "手机-0001", "品类": "手机", "品牌": "星澜", "价格": "5499元", "上市日期": "2023-01-01", "保修": "整机1年", "亮点": "影像稳定"},
        {"name": "星澜手机1代", "SKU": "手机-0073", "品类": "手机", "品牌": "星澜", "价格": "4699元", "上市日期": "2023-01-19", "保修": "整机1年", "亮点": "电池耐用"},
    ]

    report = validate_product_records(records)

    assert report["summary"]["has_issues"] is True
    assert report["issues"]["duplicate_names"][0]["name"] == "星澜手机1代"
    assert report["issues"]["duplicate_names"][0]["prices"] == ["4699元", "5499元"]


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
