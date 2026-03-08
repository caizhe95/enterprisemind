"""响应生成节点"""

import json
import re
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from graph.state import AgentState
from memory.memory_manager import get_memory_manager
from rag.evaluator import UtilityGrade
from logger import logger
from config import config

from graph.agents.common import (
    get_agent_llm,
    get_self_rag_evaluator,
    build_context,
    extract_citations,
)


def _extract_target_fields(question: str, llm) -> list[str]:
    """
    从问题中提取需要逐项回答的字段（更通用）：
    1) 多字段信号命中时，优先用 LLM 抽取 JSON 数组字段名
    2) 失败时使用规则兜底
    """
    has_multi_field_signal = any(
        k in question for k in ["分别", "各自", "分别是", "分别为", "以及", "和", "、"]
    )
    if not has_multi_field_signal:
        return []

    text_first = _extract_fields_by_text(question)
    if len(text_first) >= 2:
        return text_first

    prompt = (
        "从下面问题中提取'需要逐项回答的字段名'，仅返回JSON数组字符串。\n"
        "要求：\n"
        "1. 只保留字段名，不要实体名或修饰词。\n"
        "2. 若字段不足2个，返回空数组[]。\n"
        '3. 示例输出: ["上市时间","价格","续航"]\n\n'
        f"问题：{question}\n"
        "JSON："
    )

    try:
        raw = llm.invoke(prompt).content.strip()
        match = re.search(r"\[[\s\S]*\]", raw)
        if match:
            arr = json.loads(match.group(0))
            if isinstance(arr, list):
                cleaned = []
                for item in arr:
                    s = str(item).strip().strip("，。；;,.!?！？")
                    if not s:
                        continue
                    if len(s) > 20:
                        continue
                    if _is_placeholder_field(s):
                        continue
                    cleaned.append(s)
                cleaned = list(dict.fromkeys(cleaned))
                if len(cleaned) >= 2:
                    return cleaned[:8]
    except Exception:
        pass

    # 规则兜底（不依赖固定字段词库）
    fallback = question
    fallback = re.sub(r"[？?。！!]", "", fallback)
    # 常见模板截断，优先取“分别/以及”附近的字段片段
    for marker in ["分别是什么", "分别是", "分别为", "有哪些", "是什么"]:
        if marker in fallback:
            fallback = fallback.split(marker)[0]
            break
    parts = re.split(r"[、,，/]|以及|和|及|与", fallback)
    parts = [p.strip() for p in parts if p.strip()]
    # 去掉明显实体前缀（例如“智能手表Pro上市时间” -> “上市时间”）
    reduced = []
    for p in parts:
        m = re.search(
            r"(时间|日期|价格|售价|费用|续航|电池|容量|参数|规格|型号|版本|颜色|重量|尺寸|分辨率|屏幕|性能|功能|防水)$",
            p,
        )
        reduced.append(m.group(1) if m else p[-8:])
    reduced = list(dict.fromkeys([x for x in reduced if x]))
    reduced = [x for x in reduced if not _is_placeholder_field(x)]
    return reduced[:8] if len(reduced) >= 2 else []


def _is_placeholder_field(field: str) -> bool:
    s = (field or "").strip().lower()
    return bool(re.match(r"^(字段\d*|field\d*|item\d*)$", s))


def _extract_fields_by_text(question: str) -> list[str]:
    """优先从问题文本直接抽取字段，避免LLM返回占位字段。"""
    text = re.sub(r"[？?。！!]", "", question).strip()
    for marker in [
        "分别是什么",
        "分别是",
        "分别为",
        "分别",
        "各自",
        "各是多少",
        "是多少",
    ]:
        if marker in text:
            text = text.split(marker)[0].strip()
            break

    parts = re.split(r"[、,，/]|以及|和|及|与", text)
    parts = [p.strip() for p in parts if p.strip()]
    fields = []
    for p in parts:
        # 去掉常见实体/主语前缀，保留字段尾部
        m = re.search(
            r"(上市时间|发布时间|价格|售价|费用|续航|电池|充电|参数|规格|型号|版本|颜色|重量|尺寸|分辨率|屏幕|性能|功能|防水|容量|保修|质保)$",
            p,
        )
        c = m.group(1) if m else p
        c = c.strip("：: ")
        if not c or _is_placeholder_field(c):
            continue
        if len(c) > 12:
            continue
        fields.append(c)
    fields = list(dict.fromkeys(fields))
    return fields[:8] if len(fields) >= 2 else []


def _build_messages(
    question: str, context: str, fields: list[str], strict: bool = False
):
    if fields and len(fields) >= 2:
        style = (
            "你必须严格基于提供证据回答。禁止遗漏字段。"
            if strict
            else "基于证据回答，禁止遗漏字段。"
        )
        fields_text = "、".join(fields)
        template = "，".join([f"{f}为..." for f in fields]) + "。"
        return [
            SystemMessage(
                content=(
                    f"{style} 若证据不足请写“根据现有资料无法确定”。"
                    f"必须覆盖以下字段，不得改名，不得使用“字段1/字段2”这类占位名：{fields_text}\n"
                    f"请用一句话作答，推荐格式：{template}"
                )
            ),
            HumanMessage(content=f"证据：\n{context}\n\n问题：{question}"),
        ]

    if strict:
        return [
            SystemMessage(
                content="你必须严格基于提供的证据回答，禁止引入外部知识。如果证据不足，请明确说明'根据现有资料无法确定'。"
            ),
            HumanMessage(
                content=f"证据：\n{context}\n\n问题：{question}\n注意：只允许使用上述证据中的信息。"
            ),
        ]

    return [
        SystemMessage(content="基于以下信息回答问题，如果不确定请明确说明："),
        HumanMessage(content=f"信息：\n{context}\n\n问题：{question}"),
    ]


def _backfill_missing_fields(answer: str, fields: list[str]) -> str:
    """若模型漏字段，自动补齐并统一为一句话。"""
    if not fields or len(fields) < 2:
        return answer

    raw = answer.strip()
    # 兜底：将“字段1/字段2/字段3”按顺序映射为真实字段名
    for idx, f in enumerate(fields, 1):
        raw = re.sub(rf"字段{idx}\s*[：:]", f"{f}：", raw)

    output = raw.replace("\n", "，")
    output = output.strip("，。 ")
    if not output:
        output = "根据现有资料"

    for f in fields:
        if f not in output:
            output += f"，{f}为根据现有资料无法确定"

    if "根据现有资料无法确定" not in output and "无法确定" in output:
        output = output.replace("无法确定", "根据现有资料无法确定")

    return output.rstrip("，") + "。"


def _select_docs_for_fields(
    docs: list[dict], fields: list[str], max_docs: int = 5
) -> list[dict]:
    """
    多字段问题时按字段覆盖率选证据：
    - 优先选择能覆盖“尚未覆盖字段”的文档
    - 兼顾原始检索分（rerank/rrf）
    """
    if not docs:
        return []
    if not fields or len(fields) < 2:
        return docs[:max_docs]

    selected = []
    covered = set()
    remaining = docs[:]

    while remaining and len(selected) < max_docs:
        best_idx = -1
        best_score = -1e9
        for i, d in enumerate(remaining):
            content = d.get("content", "")
            meta = d.get("metadata", {})
            gain = 0
            for f in fields:
                if f in content and f not in covered:
                    gain += 2.0
                elif f in content:
                    gain += 0.3

            base = float(meta.get("rerank_score", 0) or 0) + float(
                meta.get("rrf_score", 0) or 0
            )
            score = gain + 0.01 * base
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx < 0:
            break

        best_doc = remaining.pop(best_idx)
        selected.append(best_doc)
        c = best_doc.get("content", "")
        for f in fields:
            if f in c:
                covered.add(f)

        if all(f in covered for f in fields):
            break

    # 证据不足时补齐到 max_docs，保持稳健
    if len(selected) < max_docs and remaining:
        selected.extend(remaining[: max_docs - len(selected)])

    return selected[:max_docs]


def reflective_generate_node(state: AgentState) -> dict:
    """
    Self-RAG 反思式生成
    生成后评估支持度，如有幻觉风险则重新生成或补充检索
    """
    docs = state.get("retrieved_docs", [])
    question = state["question"]

    llm = get_agent_llm()
    self_rag_evaluator = get_self_rag_evaluator()
    fields = _extract_target_fields(question, llm)
    docs_for_answer = _select_docs_for_fields(docs, fields, max_docs=5)
    context = build_context(docs_for_answer, state.get("tool_results", []), max_docs=5)
    messages = _build_messages(question, context, fields, strict=False)

    try:
        response = llm.invoke(messages)
        initial_answer = _backfill_missing_fields(response.content, fields)

        eval_result = self_rag_evaluator.evaluate_generation(
            question, initial_answer, docs_for_answer
        )

        logger.info(
            f"[Self-RAG] 生成评估: 支持度={eval_result['support_grade']}, "
            f"有用性={eval_result['utility_grade']}"
        )

        final_answer = initial_answer
        citations = extract_citations(docs_for_answer)

        if eval_result["is_hallucination_risk"]:
            logger.warning("[Self-RAG] 检测到幻觉风险，严格基于上下文重生成")
            strict_messages = _build_messages(question, context, fields, strict=True)
            strict_response = llm.invoke(strict_messages)
            final_answer = _backfill_missing_fields(strict_response.content, fields)

        elif eval_result["utility_grade"] == UtilityGrade.NOT_USEFUL:
            if state.get("retrieval_count", 0) < 2:
                return {
                    "next_step": "adaptive_retrieve",
                    "retrieval_count": state.get("retrieval_count", 0),
                }
            final_answer += "\n\n[注：现有资料可能无法完整回答该问题]"

        if config.HITL_ENABLE_LOW_CONF_CONFIRM and (
            eval_result.get("support_grade") in {"partially_supported", "no_support"}
            or bool(eval_result.get("is_hallucination_risk"))
        ):
            return {
                "final_answer": final_answer,
                "citations": citations,
                "self_rag_eval": eval_result,
                "next_step": "hitl_low_conf_confirm",
                "hitl_request": {
                    "type": "low_confidence_answer",
                    "question": question,
                    "eval_result": eval_result,
                    "candidate_answer": final_answer,
                    "options": ["accept", "web_retry", "conservative_retry"],
                },
            }

        user_id = state.get("user_id")
        if user_id:
            memory_mgr = get_memory_manager(state["session_id"], user_id)
            memory_mgr.update_turn(question, final_answer)

        return {
            "final_answer": final_answer,
            "citations": citations,
            "messages": [AIMessage(content=final_answer)],
            "next_step": "end",
            "self_rag_eval": eval_result,
        }

    except Exception as e:
        logger.error(f"[Self-RAG] 生成失败: {e}")
        return {"final_answer": f"生成错误: {e}", "next_step": "end"}


def response_agent_node(state: AgentState) -> dict:
    """响应Agent：整合多Agent输出并完成最终回答"""
    if not config.ENABLE_SELF_RAG:
        docs = state.get("retrieved_docs", [])
        question = state["question"]
        context = build_context(docs, state.get("tool_results", []))
        llm = get_agent_llm()
        messages = [
            SystemMessage(content="基于以下信息回答问题，如果不确定请明确说明："),
            HumanMessage(content=f"信息：\n{context}\n\n问题：{question}"),
        ]
        try:
            response = llm.invoke(messages)
            final_answer = response.content
            citations = extract_citations(docs)
            user_id = state.get("user_id")
            if user_id:
                memory_mgr = get_memory_manager(state["session_id"], user_id)
                memory_mgr.update_turn(question, final_answer)
            return {
                "final_answer": final_answer,
                "citations": citations,
                "messages": [AIMessage(content=final_answer)],
                "next_step": "end",
                "self_rag_eval": None,
                "active_agent": "response_agent",
                "agent_outputs": [
                    {
                        "agent": "response_agent",
                        "has_final_answer": bool(final_answer),
                        "self_rag_eval": None,
                        "mode": "baseline_rag",
                    }
                ],
            }
        except Exception as e:
            logger.error(f"[RAG] 普通生成失败: {e}")
            return {
                "final_answer": f"生成错误: {e}",
                "next_step": "end",
                "active_agent": "response_agent",
                "agent_outputs": [
                    {
                        "agent": "response_agent",
                        "has_final_answer": False,
                        "self_rag_eval": None,
                        "mode": "baseline_rag",
                    }
                ],
            }

    result = reflective_generate_node(state)
    next_map = {
        "adaptive_retrieve": "retrieval_agent",
        "hitl_low_conf_confirm": "hitl_low_conf_confirm",
        "end": "end",
    }
    mapped_next = next_map.get(result.get("next_step"), "end")

    return {
        **result,
        "next_step": mapped_next,
        "active_agent": "response_agent",
        "agent_outputs": [
            {
                "agent": "response_agent",
                "has_final_answer": bool(result.get("final_answer")),
                "self_rag_eval": result.get("self_rag_eval"),
            }
        ],
    }
