"""Agent共享依赖与工具函数"""

import re
from typing import Any, Dict, List, Optional

from llm_factory import get_llm
from rag.evaluator import SelfRAGEvaluator

_llm = None
_self_rag_evaluator: Optional[SelfRAGEvaluator] = None


GENERIC_FIELD_MARKERS = [
    "上市时间",
    "发布时间",
    "时间",
    "日期",
    "价格",
    "售价",
    "费用",
    "多少钱",
    "续航",
    "电池",
    "充电",
    "容量",
    "参数",
    "规格",
    "型号",
    "版本",
    "颜色",
    "重量",
    "尺寸",
    "分辨率",
    "屏幕",
    "性能",
    "功能",
    "防水",
    "质保",
    "保修",
]


def _looks_like_named_entity(text: str) -> bool:
    """
    主体实体弱识别（通用）：
    - 含有英数字型号片段（如 SW-Pro-2024 / iPhone15 / X100）
    - 或“XX的参数/价格”这种主语结构
    """
    if re.search(r"[A-Za-z]+[A-Za-z0-9\-_]*\d+[A-Za-z0-9\-_]*", text):
        return True
    if re.search(r"[\u4e00-\u9fff]{2,20}的[\u4e00-\u9fff]{1,12}", text):
        return True
    return False


def _is_structured_attribute_query(raw: str, lowered: str) -> bool:
    """是否是“明确主体 + 明确参数字段”的结构化问答。"""
    field_hits = sum(1 for k in GENERIC_FIELD_MARKERS if k in raw)
    asks_value = any(
        k in raw for k in ["分别", "各自", "是多少", "是什么", "是啥", "多少", "为"]
    ) or ("?" in raw or "？" in raw)
    multi_field = (
        any(k in raw for k in ["、", "以及", "和", "及", "与"]) or field_hits >= 2
    )
    has_subject = _looks_like_named_entity(raw) or bool(
        re.search(r"[\u4e00-\u9fffA-Za-z0-9\-]{2,20}(pro|max|ultra|版|型号)?", lowered)
    )
    return has_subject and asks_value and (multi_field or field_hits >= 1)


def get_agent_llm():
    """惰性获取 LLM，避免导入阶段失败。"""
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm


def get_self_rag_evaluator() -> SelfRAGEvaluator:
    """惰性获取 Self-RAG 评估器。"""
    global _self_rag_evaluator
    if _self_rag_evaluator is None:
        _self_rag_evaluator = SelfRAGEvaluator()
    return _self_rag_evaluator


def analyze_intent(question: str) -> Dict[str, Any]:
    """统一意图分类逻辑。"""
    lowered = question.lower().strip()
    raw = question.strip()

    math_keywords = ["计算", "等于", "+", "-", "*", "/", "^", "平方", "立方", "开方"]
    exclude_keywords = [
        "多少钱",
        "多少元",
        "多少个",
        "多少天",
        "多少人",
        "多少钱一",
        "多少块",
    ]
    if any(k in lowered for k in math_keywords) and not any(
        k in lowered for k in exclude_keywords
    ):
        return {
            "intent": "calculation",
            "reason": "命中数学表达式关键词",
            "confidence": "high",
            "should_try_search": False,
            "should_try_retrieval": False,
        }

    shopping_sql_keywords = [
        "库存",
        "现货",
        "余量",
        "优惠",
        "折扣",
        "活动价",
        "券后",
        "实时价格",
        "到手价",
    ]
    if any(k in lowered for k in shopping_sql_keywords):
        return {
            "intent": "sql",
            "reason": "命中导购实时数据关键词（库存/价格/活动）",
            "confidence": "high",
            "should_try_search": False,
            "should_try_retrieval": True,
        }

    sql_keywords = [
        "销售额",
        "销量",
        "排名",
        "统计",
        "查询",
        "总计",
        "总和",
        "平均",
        "最大",
        "最小",
        "最高",
        "最低",
    ]
    if any(k in lowered for k in sql_keywords):
        return {
            "intent": "sql",
            "reason": "命中结构化分析/统计关键词",
            "confidence": "high",
            "should_try_search": False,
            "should_try_retrieval": False,
        }

    shopping_retrieval_keywords = [
        "推荐",
        "导购",
        "对比",
        "哪个好",
        "怎么选",
        "适合",
        "预算",
        "参数",
        "续航",
        "屏幕",
        "拍照",
        "耳机",
        "手机",
        "笔记本",
        "手表",
        "家电",
    ]
    if any(k in lowered for k in shopping_retrieval_keywords):
        return {
            "intent": "retrieval",
            "reason": "命中导购咨询关键词，优先商品知识检索",
            "confidence": "high",
            "should_try_search": False,
            "should_try_retrieval": True,
        }

    # 明确参数问答：更通用的结构化识别，直达企业检索，不触发模糊确认
    if _is_structured_attribute_query(raw, lowered):
        return {
            "intent": "retrieval",
            "reason": "命中结构化参数问答（主体+字段），直达企业知识检索",
            "confidence": "high",
            "should_try_search": False,
            "should_try_retrieval": True,
        }

    search_keywords = ["最新", "新闻", "趋势", "搜索", "查一下", "网上", "百度", "谷歌"]
    web_concept_keywords = [
        "agentic rag",
        "agentic",
        "rag",
        "llm",
        "langgraph",
        "langchain",
        "prompt engineering",
        "mcp",
        "a2a",
        "multi-agent",
        "multi agent",
        "what is",
        "what's",
        "define",
        "definition",
        "概念",
        "是什么",
        "什么是",
    ]
    retrieval_keywords = [
        "公司",
        "内部",
        "文档",
        "知识库",
        "手册",
        "制度",
        "流程",
        "本项目",
        "本系统",
        "enterprise",
        "仓库",
        "代码库",
        "项目里",
        "这个项目",
        "这个系统",
        "咱们",
        "我们系统",
    ]
    asks_definition = (
        lowered.startswith("什么是")
        or lowered.startswith("what is")
        or lowered.startswith("what's")
        or lowered.endswith("是什么")
    )
    mentions_web_concept = any(k in lowered for k in web_concept_keywords)
    mentions_private_scope = any(k in lowered for k in retrieval_keywords)
    mentions_search = any(k in lowered for k in search_keywords)

    if mentions_search:
        return {
            "intent": "search",
            "reason": "显式要求联网/最新信息",
            "confidence": "high",
            "should_try_search": True,
            "should_try_retrieval": mentions_private_scope,
        }

    if asks_definition and mentions_web_concept and not mentions_private_scope:
        return {
            "intent": "search",
            "reason": "通用技术概念解释，优先走网页搜索而非企业知识库",
            "confidence": "high",
            "should_try_search": True,
            "should_try_retrieval": False,
        }

    if mentions_web_concept and not mentions_private_scope:
        return {
            "intent": "search",
            "reason": "命中通用技术主题，外部知识覆盖面更合适",
            "confidence": "medium",
            "should_try_search": True,
            "should_try_retrieval": False,
        }

    return {
        "intent": "retrieval",
        "reason": "默认落入企业知识检索",
        "confidence": "medium",
        "should_try_search": asks_definition or mentions_web_concept,
        "should_try_retrieval": True,
    }


def should_fallback_to_search(question: str, docs: Optional[List[Dict]] = None) -> bool:
    """判断当前问题在证据不足时是否值得补一次网页搜索。"""
    analysis = analyze_intent(question)
    if not analysis.get("should_try_search"):
        return False

    if not docs:
        return True

    high_signal_sources = {"tavily", "web", "url"}
    for doc in docs:
        metadata = doc.get("metadata", {})
        if metadata.get("source") in high_signal_sources:
            return False
    return True


def build_context(docs: List[Dict], tool_results: List[Dict], max_docs: int = 3) -> str:
    """构建上下文（带引用标记）"""
    parts = []
    for i, doc in enumerate(docs[:max_docs], 1):
        source = (
            doc["metadata"].get("file_name")
            or doc["metadata"].get("title")
            or doc["metadata"].get("source")
            or f"来源{i}"
        )
        grade = doc["metadata"].get("retrieval_grade", "unknown")
        content = doc["content"][:400]
        parts.append(f"[{source} | 相关度: {grade}]\n{content}")

    if tool_results:
        parts.append(f"[工具结果]\n{str(tool_results)}")

    return "\n\n".join(parts)


def extract_citations(docs: List[Dict]) -> List[Dict]:
    """提取引用"""
    return [
        {
            "source": d["metadata"].get("file_name", "unknown"),
            "title": d["metadata"].get("title"),
            "grade": d["metadata"].get("retrieval_grade", "unknown"),
            "score": d["metadata"].get("rerank_score", 0),
            "url": d["metadata"].get("url"),
        }
        for d in docs[:3]
    ]
