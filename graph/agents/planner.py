"""Planning and orchestration nodes for multi-agent execution."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from graph.state import AgentState
from graph.agents.common import analyze_intent, should_fallback_to_search
from graph.agents.field_utils import (
    canonicalize_field_name,
    extract_fields_by_text,
    field_aliases_for,
    get_metric_fields,
)


def _reset_execution_artifacts() -> Dict[str, Any]:
    return {
        "retrieved_docs": [],
        "retrieval_grade": None,
        "tool_results": [],
        "worker_trace": [],
        "tool_calls": [],
        "step_results": [],
        "last_worker_output": None,
        "extraction_context": None,
        "comparison_context": None,
        "calculation_expression": None,
        "recommendation_context": None,
        "final_answer": None,
        "citations": [],
        "guardrail_result": None,
        "self_rag_eval": None,
    }


def _classify_question(question: str, intent: str) -> str:
    text = question.strip()
    if re.search(r"(推荐|买哪个|选哪个|怎么选|适合.*吗|更适合)", text):
        return "recommendation"
    if intent == "sql" or re.search(r"(合计|总计|平均|最高|最低|前\d+|排名)", text):
        return "aggregation"
    if re.search(r"(哪个更|哪款更|谁更|更贵|更便宜|差多少|相差多少|高于|低于)", text):
        return "comparison"
    if len(extract_fields_by_text(text)) >= 2:
        return "field_list"
    return "single_fact"


def _extract_comparison_entities(question: str) -> List[str]:
    text = re.sub(r"[？?。！!]", "", question).strip()
    for marker in ["哪个更", "哪款更", "谁更", "差多少", "相差多少", "更贵", "更便宜"]:
        if marker in text:
            text = text.split(marker)[0].strip()
            break
    parts = re.split(r"[和与跟]|对比|相比", text)
    entities = [p.strip(" ，,") for p in parts if p.strip()]
    cleaned: List[str] = []
    for entity in entities:
        entity = re.sub(r"^(请问|帮我|一下|下)\s*", "", entity)
        entity = entity.strip()
        if entity:
            cleaned.append(entity)
    return cleaned[:2]


def _infer_comparison_metric(question: str) -> str:
    text = question
    if any(keyword in text for keyword in ["贵", "便宜", "更高", "更低", "高于", "低于"]):
        return "价格"
    for metric in get_metric_fields():
        for alias in field_aliases_for(metric):
            if alias and alias in text:
                return canonicalize_field_name(alias)
    if "评分" in text:
        return "评分"
    return "价格"


def _build_plan(question: str, intent: str, question_type: str) -> List[Dict[str, Any]]:
    if intent == "calculation":
        return [
            {"worker": "calculation_agent", "goal": "执行计算", "input": question},
            {"worker": "response_agent", "goal": "生成最终回答"},
        ]
    if intent == "search":
        return [
            {"worker": "search_agent", "goal": "搜索外部信息", "input": question},
            {"worker": "response_agent", "goal": "生成最终回答"},
        ]
    if intent == "sql":
        return [
            {"worker": "sql_agent", "goal": "执行结构化查询", "input": question},
            {"worker": "response_agent", "goal": "生成最终回答"},
        ]

    if question_type == "comparison":
        entities = _extract_comparison_entities(question)
        metric = _infer_comparison_metric(question)
        plan: List[Dict[str, Any]] = []
        for entity in entities:
            plan.append(
                {
                    "worker": "retrieval_agent",
                    "goal": f"查询{entity}的{metric}",
                    "input": f"{entity} {metric}",
                    "entity": entity,
                    "metric": metric,
                    "expects": ["documents_found"],
                }
            )
            plan.append(
                {
                    "worker": "extraction_agent",
                    "goal": f"抽取{entity}的{metric}",
                    "input": f"{entity} {metric}",
                    "entity": entity,
                    "metric": metric,
                    "expects": ["value_found"],
                }
            )
        if len(entities) >= 2:
            plan.append(
                {
                    "worker": "calculation_agent",
                    "goal": f"比较{metric}并计算差值",
                    "metric": metric,
                    "expects": ["calculation_done"],
                }
            )
        else:
            plan.append({"worker": "retrieval_agent", "goal": "补充检索比较证据", "input": question})
        plan.append({"worker": "response_agent", "goal": "生成最终回答"})
        return plan

    if question_type == "recommendation":
        return [
            {
                "worker": "retrieval_agent",
                "goal": "检索候选商品及导购证据",
                "input": question,
                "expects": ["documents_found"],
            },
            {
                "worker": "extraction_agent",
                "goal": "抽取候选商品结构化信息",
                "input": question,
                "expects": ["candidate_products_found"],
            },
            {
                "worker": "recommendation_agent",
                "goal": "根据预算和偏好筛选推荐候选",
                "input": question,
                "expects": ["recommendation_ready"],
            },
            {"worker": "response_agent", "goal": "生成推荐回答"},
        ]

    if question_type == "field_list":
        return [
            {
                "worker": "retrieval_agent",
                "goal": "检索多字段证据",
                "input": question,
                "expects": ["documents_found"],
            },
            {
                "worker": "extraction_agent",
                "goal": "抽取多字段结构化结果",
                "input": question,
                "expects": ["field_values_found"],
            },
            {"worker": "response_agent", "goal": "按字段组织回答"},
        ]

    return [
        {
            "worker": "retrieval_agent",
            "goal": "检索相关证据",
            "input": question,
            "expects": ["documents_found"],
        },
        {
            "worker": "extraction_agent",
            "goal": "抽取结构化证据",
            "input": question,
            "expects": ["structured_data_ready"],
        },
        {"worker": "response_agent", "goal": "生成最终回答"},
    ]


def planner_node(state: AgentState) -> dict:
    question = state["question"]
    analysis = analyze_intent(question)
    hinted_intent = state.get("routing_hint")
    intent = hinted_intent or analysis["intent"]
    question_type = _classify_question(question, intent)
    plan = _build_plan(question, intent, question_type)

    return {
        **_reset_execution_artifacts(),
        "question_type": question_type,
        "execution_plan": plan,
        "plan_version": state.get("plan_version", 0) + 1,
        "current_step_index": 0,
        "worker_input": None,
        "step_retry_counts": {},
        "replan_reason": None,
        "last_worker_output": {
            "planner": {
                "intent": intent,
                "question_type": question_type,
                "plan_length": len(plan),
            }
        },
        "next_step": "orchestrator",
        "active_agent": "planner",
        "execution_trace": [
            {
                "node": "planner",
                "decision": f"intent={intent}, question_type={question_type}, plan_steps={len(plan)}",
            }
        ],
        "agent_outputs": [
            {
                "agent": "planner",
                "intent": intent,
                "question_type": question_type,
                "plan": plan,
            }
        ],
    }


def orchestrator_node(state: AgentState) -> dict:
    plan = state.get("execution_plan") or []
    idx = int(state.get("current_step_index", 0) or 0)
    if not plan or idx >= len(plan):
        return {
            "next_step": "response_agent",
            "active_agent": "orchestrator",
            "execution_trace": [{"node": "orchestrator", "decision": "计划为空或已结束，进入response_agent"}],
        }

    step = plan[idx]
    worker = step["worker"]
    if worker == "response_agent":
        return {
            "worker_input": None,
            "next_step": "response_agent",
            "active_agent": "orchestrator",
            "execution_trace": [
                {
                    "node": "orchestrator",
                    "decision": f"step={idx + 1}, dispatch=response_agent, goal={step.get('goal')}",
                }
            ],
        }

    return {
        "worker_input": step.get("input"),
        "next_step": worker,
        "active_agent": "orchestrator",
        "execution_trace": [
            {
                "node": "orchestrator",
                "decision": f"step={idx + 1}, dispatch={worker}, goal={step.get('goal')}",
            }
        ],
    }


def replanner_node(state: AgentState) -> dict:
    plan = list(state.get("execution_plan") or [])
    idx = int(state.get("current_step_index", 0) or 0)
    if not plan or idx >= len(plan):
        return {
            "next_step": "response_agent",
            "active_agent": "replanner",
            "execution_trace": [{"node": "replanner", "decision": "无可重规划步骤，进入response_agent"}],
        }

    step = dict(plan[idx])
    retry_counts = dict(state.get("step_retry_counts") or {})
    key = str(idx)
    attempts = int(retry_counts.get(key, 0))
    question = state["question"]
    reason = state.get("replan_reason") or "当前步骤未满足成功条件"

    if step.get("worker") == "retrieval_agent":
        if reason == "fallback_to_search_due_to_low_relevance_or_empty_extraction":
            step["worker"] = "search_agent"
            step["goal"] = f"{step.get('goal', '检索相关证据')}（切换外部搜索补证据）"
            step["input"] = step.get("input") or question
            step["expects"] = ["documents_found"]
            decision = "retrieval_direct_fallback_to_search"
        elif attempts == 0:
            step["input"] = f"{step.get('input') or question} {question}".strip()
            step["goal"] = f"{step.get('goal', '检索相关证据')}（扩展查询重试）"
            decision = "retrieval_step_retry_with_expanded_query"
        elif attempts == 1:
            step["worker"] = "search_agent"
            step["goal"] = f"{step.get('goal', '检索相关证据')}（切换外部搜索补证据）"
            step["input"] = step.get("input") or question
            step["expects"] = ["documents_found"]
            decision = "retrieval_fallback_to_search"
        else:
            return {
                "next_step": "response_agent",
                "active_agent": "replanner",
                "execution_trace": [
                    {
                        "node": "replanner",
                        "decision": f"step={idx + 1} retried_exhausted，结束到response_agent | reason={reason}",
                    }
                ],
            }
    elif step.get("worker") == "calculation_agent" and state.get("comparison_context"):
        if attempts == 0:
            values = state["comparison_context"].get("values") or []
            if len(values) >= 2:
                left, right = values[0], values[1]
                high = max(int(left["value"]), int(right["value"]))
                low = min(int(left["value"]), int(right["value"]))
                step["input"] = f"{high} - {low}"
                decision = "calculation_retry_with_normalized_expression"
            else:
                decision = "calculation_retry_without_values"
        else:
            return {
                "next_step": "response_agent",
                "active_agent": "replanner",
                "execution_trace": [
                    {
                        "node": "replanner",
                        "decision": f"step={idx + 1} calculation_retry_exhausted，结束到response_agent | reason={reason}",
                    }
                ],
            }
    elif step.get("worker") == "extraction_agent":
        if reason == "fallback_to_search_due_to_low_relevance_or_empty_extraction":
            plan[idx] = {
                "worker": "search_agent",
                "goal": "外部搜索补充事实证据",
                "input": question,
                "expects": ["documents_found"],
            }
            needs_followup_extraction = state.get("question_type") in {"single_fact", "field_list"}
            next_worker = plan[idx + 1].get("worker") if idx + 1 < len(plan) else None
            if needs_followup_extraction and next_worker != "extraction_agent":
                plan[idx + 1 : idx + 1] = [
                    {
                        "worker": "extraction_agent",
                        "goal": "从外部搜索结果抽取结构化证据",
                        "input": question,
                        "expects": ["structured_data_ready"],
                    }
                ]
            elif idx + 1 >= len(plan):
                plan.append({"worker": "response_agent", "goal": "生成最终回答"})
            retry_counts[key] = attempts + 1
            return {
                "execution_plan": plan,
                "step_retry_counts": retry_counts,
                "replan_count": int(state.get("replan_count", 0) or 0) + 1,
                "replan_reason": None,
                "worker_input": None,
                "last_worker_output": None,
                "tool_results": [],
                "tool_calls": [],
                "next_step": "orchestrator",
                "active_agent": "replanner",
                "execution_trace": [
                    {
                        "node": "replanner",
                        "decision": f"step={idx + 1} extraction_fallback_to_search | reason={reason}",
                    }
                ],
                "agent_outputs": [
                    {
                        "agent": "replanner",
                        "step_index": idx,
                        "decision": "extraction_fallback_to_search",
                        "retry_count": retry_counts[key],
                    }
                ],
            }
        return {
            "next_step": "response_agent",
            "active_agent": "replanner",
            "execution_trace": [
                {
                    "node": "replanner",
                    "decision": f"step={idx + 1} extraction_retry_not_supported，结束到response_agent | reason={reason}",
                }
            ],
        }
    else:
        return {
            "next_step": "response_agent",
            "active_agent": "replanner",
            "execution_trace": [
                {
                    "node": "replanner",
                    "decision": f"step={idx + 1} 无适用重规划策略，结束到response_agent | reason={reason}",
                }
            ],
        }

    plan[idx] = step
    retry_counts[key] = attempts + 1
    return {
        "execution_plan": plan,
        "step_retry_counts": retry_counts,
        "replan_count": int(state.get("replan_count", 0) or 0) + 1,
        "replan_reason": None,
        "worker_input": None,
        "last_worker_output": None,
        "tool_results": [],
        "tool_calls": [],
        "next_step": "orchestrator",
        "active_agent": "replanner",
        "execution_trace": [
            {
                "node": "replanner",
                "decision": f"step={idx + 1} {decision} | reason={reason}",
            }
        ],
        "agent_outputs": [
            {
                "agent": "replanner",
                "step_index": idx,
                "decision": decision,
                "retry_count": retry_counts[key],
            }
        ],
    }


def _extract_numeric_metric_value(docs: List[Dict[str, Any]], metric: str) -> int | None:
    canonical_metric = canonicalize_field_name(metric)
    aliases = field_aliases_for(canonical_metric) or [canonical_metric]
    alias_pattern = "|".join(re.escape(alias) for alias in aliases if alias)
    for doc in docs:
        content = doc.get("content", "")
        if canonical_metric == "价格":
            pattern = rf"(?:{alias_pattern})\s*[:：]?\s*(\d+)\s*元?"
        else:
            pattern = rf"(?:{alias_pattern})\s*[:：]?\s*(\d+)"
        match = re.search(pattern, content)
        if match:
            return int(match.group(1))
    return None


def _extract_value_from_worker_output(worker_output: Dict[str, Any], metric: str) -> int | None:
    artifacts = worker_output.get("artifacts") or {}
    metrics = artifacts.get("metrics") or {}
    metric_payload = metrics.get(metric)
    if isinstance(metric_payload, dict) and metric_payload.get("value") is not None:
        try:
            return int(metric_payload["value"])
        except (TypeError, ValueError):
            return None

    docs = artifacts.get("retrieved_docs") or worker_output.get("retrieved_docs") or []
    return _extract_numeric_metric_value(docs, metric)


def _build_comparison_expression(state: AgentState) -> dict | None:
    plan = state.get("execution_plan") or []
    results = state.get("step_results") or []
    extraction_steps = [step for step in plan if step.get("worker") == "extraction_agent"]
    if len(extraction_steps) < 2 or len(results) < 2:
        return None

    compared = []
    for step in extraction_steps[:2]:
        entity = step.get("entity")
        metric = step.get("metric", "价格")
        match_result = next(
            (
                item
                for item in results
                if item.get("worker") == "extraction_agent" and item.get("worker_input") == step.get("input")
            ),
            None,
        )
        if not match_result:
            return None
        value = _extract_value_from_worker_output(match_result, metric)
        if value is None:
            return None
        compared.append(
            {
                "entity": entity,
                "metric": metric,
                "value": value,
                "unit": "元" if canonicalize_field_name(metric) == "价格" else None,
                "worker_input": step.get("input"),
            }
        )

    left, right = compared
    if left["value"] >= right["value"]:
        winner = left["entity"]
        expression = f"{left['value']} - {right['value']}"
    else:
        winner = right["entity"]
        expression = f"{right['value']} - {left['value']}"

    return {
        "expression": expression,
        "winner": winner,
        "metric": left["metric"],
        "values": compared,
    }


def judge_node(state: AgentState) -> dict:
    plan = state.get("execution_plan") or []
    idx = int(state.get("current_step_index", 0) or 0)
    if not plan or idx >= len(plan):
        return {"next_step": "response_agent", "active_agent": "judge"}

    step = plan[idx]
    last_output = state.get("last_worker_output") or {}
    step_result = {
        "step_index": idx,
        "worker": step.get("worker"),
        "goal": step.get("goal"),
        "worker_input": state.get("worker_input"),
        **last_output,
    }
    combined_step_results = list(state.get("step_results") or []) + [step_result]

    next_index = idx + 1
    updates: Dict[str, Any] = {
        "current_step_index": next_index,
        "step_results": [step_result],
        "active_agent": "judge",
        "execution_trace": [
            {
                "node": "judge",
                "decision": f"完成 step={idx + 1} worker={step.get('worker')}，准备进入下一步",
            }
        ],
        "next_step": "orchestrator",
    }

    worker_status = last_output.get("status", "success")
    signals = set(last_output.get("signals") or [])
    expects = set(step.get("expects") or [])
    unmet_expects = sorted(expects - signals)
    should_try_search_fallback = False

    if step.get("worker") == "retrieval_agent":
        docs = (last_output.get("artifacts") or {}).get("retrieved_docs") or state.get("retrieved_docs") or []
        retrieval_grade = (last_output.get("artifacts") or {}).get("retrieval_grade")
        should_try_search_fallback = bool(
            should_fallback_to_search(state["question"], docs)
            and (
                not docs
                or retrieval_grade == "irrelevant"
                or (
                    state.get("question_type") in {"single_fact", "field_list"}
                    and retrieval_grade != "highly_relevant"
                )
            )
        )
    elif step.get("worker") == "extraction_agent":
        artifacts = last_output.get("artifacts") or {}
        has_structured = bool(
            artifacts.get("fields") or artifacts.get("metrics") or artifacts.get("products")
        )
        docs = state.get("retrieved_docs") or []
        should_try_search_fallback = bool(
            not has_structured and should_fallback_to_search(state["question"], docs)
        )

    if worker_status == "failed" or unmet_expects:
        updates["current_step_index"] = idx
        updates["next_step"] = "replanner"
        updates["replan_reason"] = (
            f"worker_status={worker_status}, unmet_expects={','.join(unmet_expects) or 'none'}"
        )
        updates["execution_trace"] = [
            {
                "node": "judge",
                "decision": f"step={idx + 1} worker={step.get('worker')} 未满足条件，进入replanner",
            }
        ]
    elif should_try_search_fallback:
        updates["current_step_index"] = idx
        updates["next_step"] = "replanner"
        updates["replan_reason"] = "fallback_to_search_due_to_low_relevance_or_empty_extraction"
        updates["execution_trace"] = [
            {
                "node": "judge",
                "decision": f"step={idx + 1} worker={step.get('worker')} 本地证据不足，切换外部搜索补证据",
            }
        ]

    if state.get("question_type") == "comparison" and step.get("worker") == "extraction_agent":
        comparison_payload = _build_comparison_expression(
            {**state, **updates, "step_results": combined_step_results}
        )
        if comparison_payload:
            values = comparison_payload["values"]
            updates["comparison_context"] = {
                "metric": comparison_payload["metric"],
                "winner": comparison_payload["winner"],
                "values": values,
            }
            updates["calculation_expression"] = comparison_payload["expression"]
            updates["tool_results"] = [
                {
                    "tool": "comparison_prepare",
                    "result": {
                        "winner": comparison_payload["winner"],
                        "metric": comparison_payload["metric"],
                        "values": values,
                        "expression": comparison_payload["expression"],
                    },
                }
            ]

    if next_index >= len(plan):
        updates["next_step"] = "response_agent"

    return updates
