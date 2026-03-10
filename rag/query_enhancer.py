"""智能查询增强器 - 按需启用策略"""

import re
import time
from typing import List, Dict
from llm_factory import get_llm
from logger import logger


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
        # 企业场景同义词库（可扩展为配置文件加载）
        self.synonyms = {
            "销售额": ["营收", "收入", "销售金额", "营业额"],
            "产品": ["商品", "货物", "SKU", "型号", "Item"],
            "客户": ["用户", "顾客", "购买者", "消费者", "买家"],
            "价格": ["单价", "售价", "定价", "多少钱", "费用"],
            "库存": ["存货", "仓储", "备货量", "存储量"],
            "优惠": ["折扣", "促销", "活动价", "减免", "让利"],
            "排名": ["排序", "前几名", "Top", "最佳", "最受欢迎"],
            "保修": ["质保", "维修", "售后保障", "保修政策"],
            "发票": ["电子发票", "纸质发票", "开票", "发票抬头"],
            "配送": ["发货", "物流", "到货", "配送时效"],
            "会员": ["会员等级", "会员权益", "折扣等级", "会员规则"],
            "渠道": ["线上", "线下", "销售渠道", "购买渠道"],
            "销量": ["销售量", "卖出数量", "出货量", "成交量"],
            "退换货": ["退货", "换货", "售后政策", "退换政策"],
            "价保": ["价格保护", "保价", "补差", "价保政策"],
            "活动": ["促销活动", "优惠活动", "限时活动", "大促"],
        }

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
            "is_complex": any(
                w in normalized_query
                for w in ["和", "对比", "vs", "以及", "同时", "分别", "区别"]
            ),
            "has_synonyms": any(k in normalized_query for k in self.synonyms.keys()),
        }

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

        for key, variants in self.synonyms.items():
            if key in query:
                for variant in variants:
                    new_query = query.replace(key, variant)
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
            parts = [
                p.strip(" ，。？?")
                for p in re.split(sep_pattern, left)
                if p.strip(" ，。？?")
            ]
            if len(parts) >= 2:
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
        tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9\-]{2,20}", text)
        return [t for t in tokens if t not in stopwords]

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
