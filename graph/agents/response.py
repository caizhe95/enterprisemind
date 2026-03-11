"""响应生成节点"""

import json
import re
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from graph.state import AgentState
from memory.memory_manager import get_memory_manager
from logger import logger
from config import config

from graph.agents.common import (
    get_agent_llm,
    get_self_rag_evaluator,
    build_context,
    extract_citations,
)
from graph.agents.field_utils import (
    canonicalize_field_name,
    extract_fields_by_text,
    field_aliases_for,
    has_explicit_field_list_signal,
    is_placeholder_field,
    normalize_candidate_field,
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

    text_first = extract_fields_by_text(question)
    if len(text_first) >= 2:
        return text_first
    if not has_explicit_field_list_signal(question):
        return []

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
                    if is_placeholder_field(s):
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
    reduced: list[str] = []
    for p in parts:
        normalized = normalize_candidate_field(p)
        if normalized:
            reduced.append(normalized)
            continue
        reduced.append(canonicalize_field_name(p[-8:]))
    reduced = list(dict.fromkeys([x for x in reduced if x]))
    reduced = [x for x in reduced if not is_placeholder_field(x)]
    return reduced[:8] if len(reduced) >= 2 else []


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


def _format_metric_value(metric: str, value: object) -> str:
    if value is None:
        return "根据现有资料无法确定"
    if canonicalize_field_name(metric) == "价格":
        return f"{value}元"
    return str(value)


def _build_structured_comparison_answer(state: AgentState) -> str | None:
    context = state.get("comparison_context") or {}
    values = context.get("values") or []
    winner = context.get("winner")
    metric = context.get("metric") or "价格"
    if not winner or len(values) < 2:
        return None

    calc_result = None
    for item in reversed(state.get("step_results") or []):
        if item.get("worker") == "calculation_agent":
            calc_result = item
            break

    difference = None
    if calc_result:
        artifacts = calc_result.get("artifacts") or {}
        tool_results = artifacts.get("tool_results") or []
        for tool_result in tool_results:
            text = str(tool_result.get("result") or "")
            match = re.search(r"=\s*(-?\d+(?:\.\d+)?)", text)
            if match:
                raw_num = float(match.group(1))
                difference = int(raw_num) if raw_num.is_integer() else raw_num
                break

    if difference is None:
        numeric_values = [int(v["value"]) for v in values if v.get("value") is not None]
        if len(numeric_values) >= 2:
            difference = max(numeric_values) - min(numeric_values)

    if difference is None:
        return None

    ordered_values = sorted(values, key=lambda item: item.get("value", 0), reverse=True)
    leader = ordered_values[0]
    trailer = ordered_values[1] if len(ordered_values) > 1 else None
    metric_label = canonicalize_field_name(metric)
    if metric_label == "价格" and trailer:
        return (
            f"{winner}更贵，{leader['entity']}价格为{leader['value']}元，"
            f"{trailer['entity']}价格为{trailer['value']}元，贵{difference}元。"
        )
    if metric == "价格":
        return f"{winner}更贵，贵{difference}元。"
    return f"{winner}的{metric}更高，相差{difference}。"


def _build_structured_field_list_answer(state: AgentState) -> str | None:
    question = state["question"]
    fields = extract_fields_by_text(question)
    if len(fields) < 2:
        return None

    value_map: dict[str, str] = {}
    for item in state.get("step_results") or []:
        if item.get("worker") != "extraction_agent":
            continue
        artifacts = item.get("artifacts") or {}
        artifact_fields = artifacts.get("fields") or {}
        metrics = artifacts.get("metrics") or {}

        for field in fields:
            if field in value_map:
                continue
            if field in artifact_fields:
                value_map[field] = str(artifact_fields[field])
                continue
            metric_payload = metrics.get(field)
            if isinstance(metric_payload, dict) and metric_payload.get("value") is not None:
                value_map[field] = _format_metric_value(field, metric_payload["value"])
                continue

    if not value_map:
        return None

    parts = [
        f"{field}为{value_map.get(field, '根据现有资料无法确定')}"
        for field in fields
    ]
    return "，".join(parts) + "。"


def _build_structured_single_fact_answer(state: AgentState) -> str | None:
    for item in reversed(state.get("step_results") or []):
        if item.get("worker") != "extraction_agent":
            continue
        artifacts = item.get("artifacts") or {}
        fields = artifacts.get("fields") or {}
        metrics = artifacts.get("metrics") or {}
        if fields:
            field, value = next(iter(fields.items()))
            return f"{field}为{value}。"
        if metrics:
            field, payload = next(iter(metrics.items()))
            return f"{field}为{_format_metric_value(field, payload.get('value'))}。"
    return None


def _try_build_structured_answer(state: AgentState) -> str | None:
    question_type = state.get("question_type")
    if question_type == "comparison":
        return _build_structured_comparison_answer(state)
    if question_type == "field_list":
        return _build_structured_field_list_answer(state)
    if question_type == "single_fact":
        return _build_structured_single_fact_answer(state)
    if question_type == "recommendation":
        context = state.get("recommendation_context") or {}
        recommendations = context.get("recommendations") or []
        if recommendations:
            top = recommendations[0]
            reasons = "、".join(top.get("reasons") or []) or "整体更符合当前需求"
            answer = f"更推荐{top['name']}"
            if top.get("price") is not None:
                answer += f"，价格约{top['price']}元"
            answer += f"，原因是{reasons}"
            coverage_gaps = context.get("coverage_gaps") or []
            if coverage_gaps:
                answer += f"；注意：{coverage_gaps[0]}"
            if len(recommendations) > 1:
                backup = recommendations[1]
                answer += f"；备选可以看{backup['name']}"
                backup_reasons = "、".join(backup.get("reasons") or [])
                if backup_reasons:
                    answer += f"，它的优势是{backup_reasons}"
            return answer + "。"
    return None


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
            f"幻觉风险={eval_result['is_hallucination_risk']}"
        )

        final_answer = initial_answer
        citations = extract_citations(docs_for_answer)

        if eval_result["is_hallucination_risk"] and config.ENABLE_SELF_RAG_GUARD:
            logger.warning("[Self-RAG] 检测到幻觉风险，严格基于上下文重生成")
            strict_messages = _build_messages(question, context, fields, strict=True)
            strict_response = llm.invoke(strict_messages)
            final_answer = _backfill_missing_fields(strict_response.content, fields)

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
            "guardrail_result": eval_result,
        }

    except Exception as e:
        logger.error(f"[Self-RAG] 生成失败: {e}")
        return {"final_answer": f"生成错误: {e}", "next_step": "end"}


def response_agent_node(state: AgentState) -> dict:
    """响应Agent：整合多Agent输出并完成最终回答"""
    structured_answer = _try_build_structured_answer(state)
    if structured_answer:
        citations = extract_citations(state.get("retrieved_docs", []))
        return {
            "final_answer": structured_answer,
            "citations": citations,
            "messages": [AIMessage(content=structured_answer)],
            "next_step": "end",
            "self_rag_eval": None,
            "active_agent": "response_agent",
            "agent_outputs": [
                {
                    "agent": "response_agent",
                    "has_final_answer": True,
                    "self_rag_eval": None,
                    "mode": "structured_synthesis",
                }
            ],
        }

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
    if state.get("execution_plan") and result.get("next_step") == "retrieval_agent":
        result["next_step"] = "end"
    next_map = {
        "retrieval_agent": "retrieval_agent",
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
