"""智能查询增强器 - 按需启用策略"""

import re
import time
from typing import List, Dict
from llm_factory import get_llm
from logger import logger
from graph.agents.field_utils import get_query_synonym_groups
from graph.agents.section_utils import get_section_synonym_groups


class SmartQueryEnhancer:
    """
    智能查询增强器
    策略：规则扩展（必选）+ HyDE（条件触发）+ 查询拆解（复杂场景）
    避免过度增强造成成本浪费
    """

    def __init__(self):
        self.llm = get_llm()
        self.no_decompose_tag = "__NODECOMP__"
        self.decompose_cache_ttl = 1800  # 30分钟
        self.decompose_cache: Dict[str, Dict] = {}
        self.synonyms = self._load_synonyms()
        self.synonym_lookup = self._build_synonym_lookup(self.synonyms)

    @staticmethod
    def _load_synonyms() -> Dict[str, List[str]]:
        groups = list(get_query_synonym_groups()) + list(get_section_synonym_groups())
        synonym_map: Dict[str, List[str]] = {}
        for group in groups:
            canonical = str(group[0]).strip()
            variants = [str(item).strip() for item in group[1:] if str(item).strip()]
            if canonical:
                synonym_map[canonical] = variants
        return synonym_map

    @staticmethod
    def _build_synonym_lookup(synonyms: Dict[str, List[str]]) -> Dict[str, str]:
        lookup: Dict[str, str] = {}
        for canonical, variants in synonyms.items():
            terms = [canonical, *variants]
            for term in terms:
                normalized = str(term).strip()
                if normalized and normalized not in lookup:
                    lookup[normalized] = canonical
        return lookup

    def analyze_query(self, query: str) -> Dict[str, bool]:
        """
        查询特征分析，决定启用哪些增强策略
        返回: {use_hyde, use_expansion, use_decompose}
        """
        no_decompose = query.startswith(self.no_decompose_tag)
        normalized_query = (
            query[len(self.no_decompose_tag) :].strip() if no_decompose else query
        )

        features = {
            "length": len(normalized_query),
            "is_short": len(normalized_query) < 15,
            "is_question": any(
                w in normalized_query
                for w in ["吗", "?", "？", "哪", "什么", "如何", "怎么", "为什么"]
            ),
            "is_abstract": any(
                w in normalized_query
                for w in ["哪个", "最佳", "最好", "推荐", "最优", "合适"]
            ),
            "is_concrete": any(
                w in normalized_query
                for w in ["价格", "库存", "销量", "2024", "2025", "具体", "数据"]
            ),
            "is_compare": any(
                w in normalized_query
                for w in ["对比", "vs", "区别", "哪个更", "哪个贵", "哪个便宜"]
            ),
            "has_list_intent": any(
                w in normalized_query
                for w in ["分别", "各自", "分别是", "分别为", "分别是什么", "有哪些"]
            ),
            "has_multi_clause_connector": any(
                w in normalized_query for w in ["以及", "同时", "并且"]
            )
            or (
                "和" in normalized_query
                and any(
                    w in normalized_query
                    for w in ["什么", "哪些", "多少", "价格", "品类", "参数", "配置", "政策"]
                )
            ),
            "has_synonyms": any(
                term in normalized_query for term in self.synonym_lookup.keys()
            ),
        }
        features["is_complex"] = (
            (features["has_list_intent"] or features["has_multi_clause_connector"])
            and not features["is_compare"]
        )

        # 智能决策逻辑
        strategy = {
            # HyDE：短查询 + 抽象疑问句（需要补充具体信息）
            "use_hyde": (
                features["is_short"]
                and features["is_question"]
                and features["is_abstract"]
                and not features["is_concrete"]
            ),
            # 扩展：包含业务同义词且查询不太长（避免长尾查询爆炸）
            "use_expansion": (
                features["has_synonyms"]
                and features["length"] < 30
                and not features["is_complex"]  # 复杂查询先拆解再扩展
            ),
            # 拆解：明显包含多个子问题且长度足够
            "use_decompose": (not no_decompose)
            and features["is_complex"]
            and not features["is_compare"]
            and features["length"] > 10,
        }

        logger.info(
            f"[QueryAnalysis] 查询'{normalized_query[:20]}...' 特征: {features} → 策略: {strategy}"
        )
        return strategy

    def enhance(self, query: str) -> List[str]:
        """
        按需增强，避免无效调用
        返回查询变体列表（原始查询 + 增强变体）
        """
        no_decompose = query.startswith(self.no_decompose_tag)
        base_query = (
            query[len(self.no_decompose_tag) :].strip() if no_decompose else query
        )
        strategies = self.analyze_query(query)
        queries = [base_query]  # 原始查询必须保留

        # 策略1: 查询拆解（最高优先级，改变查询结构）
        if strategies["use_decompose"]:
            sub_queries = self._decompose_query(base_query)
            # 拆解后对每个子查询做简单扩展，不做HyDE（避免递归爆炸）
            for sq in sub_queries:
                sq_text = self._sanitize_subquery_text(sq.get("query", ""))
                if not sq_text:
                    continue
                expanded = self._rule_expand(sq_text)
                queries.extend(expanded)
            final_queries = list(dict.fromkeys(queries))[:5]  # 去重并限制
            logger.info(f"[Decompose] 实际检索变体: {final_queries}")
            return final_queries

        # 策略2: 规则扩展（零成本，次优先级）
        if strategies["use_expansion"]:
            expanded = self._rule_expand(base_query)
            queries.extend([q for q in expanded if q != base_query])

        # 策略3: HyDE（有成本，最后执行，且只生成一个增强版本）
        if strategies["use_hyde"]:
            hyde_query = self._hyde_enhance(base_query)
            if hyde_query != base_query:
                queries.append(hyde_query)
                logger.info(f"[HyDE] 生成增强查询: {hyde_query[:50]}...")

        # 去重并限制数量（防止召回过多影响性能）
        unique_queries = list(dict.fromkeys(queries))
        return unique_queries[:4]  # 最多4个变体

    def _rule_expand(self, query: str) -> List[str]:
        """基于规则的同义词扩展（零成本）"""
        expansions = [query]
        matched_terms = [
            term
            for term in sorted(self.synonym_lookup.keys(), key=len, reverse=True)
            if term in query
        ]

        for term in matched_terms:
            canonical = self.synonym_lookup[term]
            related_terms = [canonical] + [
                variant
                for variant in self.synonyms.get(canonical, [])
                if variant and variant != term
            ]
            for replacement in related_terms:
                if replacement == term:
                    continue
                new_query = query.replace(term, replacement)
                if new_query not in expansions:
                    expansions.append(new_query)
                    if len(expansions) >= 3:  # 限制扩展数量
                        return expansions

        # 口语化变体转换（疑问句↔陈述句）
        if "吗" in query or "?" in query or "？" in query:
            statement = (
                query.replace("吗", "").replace("?", "").replace("？", "").strip()
            )
            if statement and statement not in expansions:
                expansions.append(statement)

        return expansions

    def _hyde_enhance(self, query: str) -> str:
        """
        HyDE：生成假设答案作为检索桥梁
        解决语义鸿沟（查询与文档表述不一致）
        """
        try:
            prompt = f"""基于常识，简要回答以下问题（1-2句话），用于帮助检索相关文档。
要求：回答要具体、包含可能的实体名称，不要泛泛而谈。

问题：{query}
回答："""

            response = self.llm.invoke(prompt).content.strip()
            enhanced = f"{query} {response}"

            return enhanced

        except Exception as e:
            logger.error(f"[HyDE] 增强失败: {e}")
            return query

    def _decompose_query(self, query: str) -> List[Dict]:
        """复杂查询拆解为子查询"""
        cached = self.decompose_cache.get(query)
        now = time.time()
        if cached and now - cached["ts"] < self.decompose_cache_ttl:
            logger.info(f"[Decompose] 命中缓存: {len(cached['items'])} 个子查询")
            return cached["items"]

        # 先规则拆解，只有失败再调用LLM
        rule_sub_queries = self._rule_decompose_query(query)
        if len(rule_sub_queries) >= 2:
            validated = self._validate_sub_queries(query, rule_sub_queries)
            if len(validated) >= 2:
                logger.info(f"[Decompose] 规则拆解为 {len(validated)} 个子查询")
                self.decompose_cache[query] = {"ts": now, "items": validated}
                return validated

        try:
            prompt = f"""将以下复合问题拆分为2-3个独立子问题，用JSON数组返回：
问题：{query}

要求：
1. 每个子问题应能独立回答
2. 保留原始问题中的实体（时间、地点、产品名）
3. 格式：[{{"query": "子问题1", "intent": "查询意图"}}]

JSON："""

            response = self.llm.invoke(prompt).content
            import json

            match = re.search(r"\[.*\]", response, re.DOTALL)
            if match:
                raw_sub_queries = json.loads(match.group())
                normalized = []
                for item in raw_sub_queries:
                    if not isinstance(item, dict):
                        continue
                    q = self._sanitize_subquery_text(str(item.get("query", "")).strip())
                    if not q:
                        continue
                    normalized.append(
                        {
                            "query": q,
                            "intent": str(item.get("intent", "sub_query")).strip()
                            or "sub_query",
                        }
                    )

                if normalized:
                    validated = self._validate_sub_queries(query, normalized)
                    if len(validated) >= 2:
                        logger.info(f"[Decompose] 拆解为 {len(validated)} 个子查询")
                        logger.info(
                            f"[Decompose] 子查询详情: {[x['query'] for x in validated]}"
                        )
                        self.decompose_cache[query] = {"ts": now, "items": validated}
                        return validated
        except Exception as e:
            logger.error(f"[Decompose] 拆解失败: {e}")

        return [{"query": query, "intent": "original"}]

    def _rule_decompose_query(self, query: str) -> List[Dict]:
        """规则优先拆解，覆盖大部分“分别/以及/和/对比”场景。"""
        base = query.strip()
        if not base:
            return [{"query": query, "intent": "original"}]

        sep_pattern = r"[、,，/]|以及|和|及|与"
        field_parts = re.split(sep_pattern, base)
        field_parts = [p.strip(" ，。？?") for p in field_parts if p.strip(" ，。？?")]

        marker = None
        for m in ["分别是什么", "分别是", "分别为", "分别", "有哪些", "是什么"]:
            if m in base:
                marker = m
                break

        if marker:
            left = base.split(marker)[0].strip()
            entity_prefix = ""
            field_segment = left
            if "的" in left:
                entity_prefix, field_segment = left.split("的", 1)
                entity_prefix = entity_prefix.strip(" ，。？?")
                field_segment = field_segment.strip(" ，。？?")
            parts = [
                p.strip(" ，。？?")
                for p in re.split(sep_pattern, field_segment)
                if p.strip(" ，。？?")
            ]
            if len(parts) >= 2:
                if entity_prefix:
                    return [
                        {"query": f"{entity_prefix}的{p}", "intent": "sub_query"}
                        for p in parts[:3]
                    ]
                return [{"query": p, "intent": "sub_query"} for p in parts[:3]]

        if "对比" in base or "区别" in base:
            if len(field_parts) >= 2:
                return [
                    {"query": p, "intent": "compare_aspect"} for p in field_parts[:3]
                ]

        return [{"query": query, "intent": "original"}]

    def _validate_sub_queries(
        self, original_query: str, sub_queries: List[Dict]
    ) -> List[Dict]:
        """拆解质量校验：去重、长度校验、实体锚点校验。"""
        if not sub_queries:
            return [{"query": original_query, "intent": "original"}]

        anchors = self._extract_anchor_tokens(original_query)
        primary_anchor = anchors[0] if anchors else ""

        cleaned = []
        seen = set()
        for item in sub_queries:
            if not isinstance(item, dict):
                continue
            q = self._sanitize_subquery_text(str(item.get("query", "")).strip())
            if not q:
                continue
            if len(q) < 3 or len(q) > 120:
                continue

            if anchors and not any(a in q for a in anchors[:3]) and primary_anchor:
                q = f"{primary_anchor} {q}"

            if q in seen:
                continue
            seen.add(q)
            cleaned.append(
                {
                    "query": q,
                    "intent": str(item.get("intent", "sub_query")).strip()
                    or "sub_query",
                }
            )

        if len(cleaned) < 2:
            return [{"query": original_query, "intent": "original"}]

        return cleaned[:3]

    @staticmethod
    def _extract_anchor_tokens(text: str) -> List[str]:
        stopwords = {
            "什么",
            "多少",
            "怎么",
            "如何",
            "请问",
            "分别",
            "以及",
            "和",
            "与",
            "及",
            "对比",
            "区别",
            "是",
            "的",
        }
        entity_candidate = text.strip()
        for marker in ["的", "和", "与", "以及", "对比", "区别", "哪个", "什么", "多少"]:
            if marker in entity_candidate:
                entity_candidate = entity_candidate.split(marker, 1)[0].strip()
                break
        if 2 <= len(entity_candidate) <= 30 and entity_candidate not in stopwords:
            return [entity_candidate]
        tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9\-]{2,20}", text)
        return [t for t in tokens if t not in stopwords and len(t) <= 12]

    @staticmethod
    def _sanitize_subquery_text(text: str) -> str:
        """清洗子查询文本，避免编号/换行污染检索。"""
        if not text:
            return ""
        cleaned = text.replace("\r", "\n").strip()
        if "\n" in cleaned:
            cleaned = cleaned.split("\n", 1)[0].strip()
        cleaned = re.sub(r"^\d+[.)、\s-]+", "", cleaned).strip()
        return cleaned[:200]


# 便捷函数
_query_enhancer = None


def get_query_enhancer() -> SmartQueryEnhancer:
    """单例获取增强器"""
    global _query_enhancer
    if _query_enhancer is None:
        _query_enhancer = SmartQueryEnhancer()
    return _query_enhancer
