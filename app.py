"""Gradio UI entrypoint."""

from typing import List, Generator
import gradio as gr
import uuid
import time
from langgraph.types import Command

from graph.builder import app
from graph.state_helpers import build_initial_state, normalize_interrupt
from memory.memory_manager import get_memory_manager, clear_memory_manager
from cache.cache_manager import cache_manager
from config import check_environment, config
from logger import logger

check_environment()

APP_CSS = """
:root {
  --bg: #f7f7f8;
  --card: #ffffff;
  --line: #e5e7eb;
  --text: #111827;
  --muted: #6b7280;
  --brand: #111827;
}

.gradio-container {
  background: var(--bg);
  color: var(--text);
  max-width: 1320px !important;
  margin: 0 auto !important;
  padding: 12px 16px 24px 16px !important;
}

.gradio-container .gr-box,
.gradio-container .panel {
  border: 1px solid var(--line) !important;
  border-radius: 14px !important;
  background: var(--card) !important;
  box-shadow: 0 1px 2px rgba(16, 24, 40, 0.06);
}

.gradio-container h1,
.gradio-container h2,
.gradio-container h3 {
  color: var(--text);
}

.gradio-container p,
.gradio-container .prose,
.gradio-container label {
  color: var(--muted);
}

#chatbot-wrap {
  border: 1px solid var(--line);
  border-radius: 14px;
  background: var(--card);
  padding: 6px;
}

#chatbot-wrap .bubble-wrap {
  max-width: 85%;
}

#input-wrap {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 8px;
}

#input-wrap .gr-textbox textarea {
  border: none !important;
  box-shadow: none !important;
  border-radius: 10px !important;
  min-height: 72px !important;
}

.self-rag-badge {
  font-size: 0.78em;
  padding: 2px 8px;
  border-radius: 999px;
  margin-left: 8px;
  border: 1px solid transparent;
}
.grade-high { background: #ecfdf3; color: #067647; border-color: #abefc6; }
.grade-partial { background: #fffaeb; color: #b54708; border-color: #fedf89; }
.grade-low { background: #fef3f2; color: #b42318; border-color: #fecdca; }

.btn-uniform button,
button.btn-uniform,
.btn-uniform {
  height: 44px !important;
  min-height: 44px !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
}

.btn-uniform button {
  width: 100%;
}
"""


def create_interface():
    with gr.Blocks(title="智能导购多 Agent 系统") as demo:
        gr.Markdown("## 智能导购多 Agent 系统")
        gr.Markdown("智能导购问答（检索、抽取、推荐、对比、SQL 分析）")

        thread_id = gr.State(lambda: str(uuid.uuid4()))
        session_id = gr.State(lambda: f"sess_{int(time.time())}")
        user_id = gr.State(lambda: f"user_{int(time.time())}")
        interrupt_payload = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=7):
                with gr.Column(elem_id="chatbot-wrap"):
                    chatbot = gr.Chatbot(height=640, show_label=False)

                with gr.Row(elem_id="input-wrap"):
                    msg_input = gr.Textbox(
                        placeholder="描述你的预算、用途和偏好，例如：预算3000，轻薄长续航笔记本",
                        show_label=False,
                        scale=9,
                        lines=3,
                    )
                    submit_btn = gr.Button("发送", variant="primary", scale=1)
                    submit_btn.elem_classes = ["btn-uniform"]

            with gr.Column(scale=3):
                with gr.Accordion("高级设置", open=True):
                    with gr.Row():
                        show_reasoning = gr.Checkbox(label="显示推理过程", value=True)
                        show_debug = gr.Checkbox(label="调试信息", value=False)

                    status_text = gr.Textbox(
                        label="执行状态", value="就绪", interactive=False, lines=2
                    )
                    route_text = gr.Textbox(
                        label="意图路由",
                        value="等待路由...",
                        interactive=False,
                        lines=2,
                    )
                    citation_md = gr.Markdown(label="引用溯源")
                    eval_json = gr.JSON(label="Self-RAG评估详情", visible=False)

                    with gr.Group(visible=False) as decision_panel:
                        strategy_choice = gr.Radio(
                            choices=[
                                "auto",
                                "search",
                                "retrieval",
                                "sql",
                                "calculation",
                            ],
                            value="auto",
                            label="问题策略",
                            info="auto=自动判断；其余为强制路由",
                        )
                        low_conf_choice = gr.Radio(
                            choices=["accept", "web_retry", "conservative_retry"],
                            value="accept",
                            label="低置信处理",
                            info="accept=接受当前答案；web_retry=补充联网重答；conservative_retry=保守重答",
                        )
                        worker_slot_input = gr.Textbox(
                            label="导购补充信息",
                            placeholder="例如：预算3000-4000，偏好轻薄和长续航",
                            lines=2,
                        )
                        decision_btn = gr.Button(
                            "应用决策",
                            variant="secondary",
                            elem_classes=["btn-uniform"],
                        )
                        hitl_json = gr.JSON(label="待确认任务")
                    retrieval_status = gr.Textbox(
                        "等待查询...", lines=3, label="检索质量"
                    )
                    gen_status = gr.Textbox("等待生成...", lines=2, label="生成验证")

                    with gr.Row():
                        memory_btn = gr.Button(
                            "查看当前记忆", elem_classes=["btn-uniform"]
                        )
                        clear_memory_btn = gr.Button(
                            "清除记忆", elem_classes=["btn-uniform"]
                        )
                    refresh_btn = gr.Button(
                        "刷新缓存统计", elem_classes=["btn-uniform"]
                    )
                    memory_json = gr.JSON(label="记忆状态")
                    cache_json = gr.JSON(label="缓存统计")

        def process_message_stream(
            message: str,
            history: List[dict],
            thread_id_val: str,
            sess_id: str,
            uid: str,
            strategy_choice: str,
            show_reasoning: bool,
            show_debug: bool,
        ) -> Generator:
            """流式处理消息，显示Self-RAG过程"""

            if not message:
                yield (
                    history,
                    "空输入",
                    "等待路由...",
                    "",
                    "",
                    "",
                    {},
                    None,
                    {},
                    gr.update(visible=False),
                )
                return

            config_dict = {"configurable": {"thread_id": thread_id_val}}
            history = (history or []) + [{"role": "user", "content": message}]
            yield (
                history,
                "🤔 正在分析意图...",
                "正在识别问题类型...",
                "初始化...",
                "等待中...",
                "",
                {},
                None,
                {},
                gr.update(visible=False),
            )

            routing_hint = None if strategy_choice == "auto" else strategy_choice
            initial_state = build_initial_state(message, sess_id, uid, routing_hint)

            yield from _stream_graph(
                payload=initial_state,
                history=history,
                config_dict=config_dict,
                sess_id=sess_id,
                uid=uid,
                current_question=message,
                show_reasoning=show_reasoning,
                show_debug=show_debug,
            )

        def _stream_graph(
            payload,
            history: List[dict],
            config_dict: dict,
            sess_id: str,
            uid: str,
            current_question: str,
            show_reasoning: bool,
            show_debug: bool,
        ) -> Generator:
            full_response = ""
            retrieval_detail = "🔄 检索中..."
            gen_detail = "等待生成..."
            cite_md = ""
            route_detail = "等待路由..."

            try:
                for event in app.stream(payload, config_dict, stream_mode="values"):
                    if isinstance(event, dict) and "__interrupt__" in event:
                        pending = normalize_interrupt(event["__interrupt__"])
                        pending_type = pending.get("type")
                        if pending_type == "strategy_confirm":
                            recommended = pending.get("recommended", "retrieval")
                            history.append(
                                {
                                    "role": "assistant",
                                    "content": (
                                        "你的问题策略不够明确。"
                                        f"\n建议策略: `{recommended}`"
                                        "\n请在【高级设置】里选择策略后点击“应用决策”。"
                                    ),
                                }
                            )
                            status = "⏸️ 等待策略确认"
                            analysis = pending.get("analysis", {})
                            source = analysis.get("route_source", "rules")
                            reason = analysis.get("reason", "待确认")
                            route_detail = f"来源: {source} | {reason}"
                        else:
                            if pending_type == "shopping_slot_confirm":
                                history.append(
                                    {
                                        "role": "assistant",
                                        "content": (
                                            "导购建议需要补充信息。"
                                            "\n请在【高级设置-导购补充信息】填写预算和偏好后点击“应用决策”。"
                                        ),
                                    }
                                )
                                status = "⏸️ 等待导购偏好补充"
                            else:
                                history.append(
                                    {
                                        "role": "assistant",
                                        "content": (
                                            "当前答案置信度偏低。"
                                            "\n请在【高级设置】选择低置信处理方式后点击“应用决策”。"
                                        ),
                                    }
                                )
                                status = "⏸️ 等待低置信决策"

                        yield (
                            history,
                            status,
                            route_detail,
                            retrieval_detail,
                            "等待用户决策...",
                            cite_md,
                            {},
                            pending,
                            pending,
                            gr.update(visible=True),
                        )
                        return

                    state = event
                    grade = state.get("retrieval_grade")
                    reflection = state.get("reflection_count", 0)
                    eval_result = state.get("self_rag_eval", {})
                    citations = state.get("citations", [])
                    agent_outputs = state.get("agent_outputs", []) or []
                    supervisor_output = next(
                        (
                            item
                            for item in reversed(agent_outputs)
                            if item.get("agent") == "supervisor"
                        ),
                        None,
                    )

                    status_parts = []
                    if supervisor_output:
                        route_source = supervisor_output.get("route_source", "rules")
                        route_intent = supervisor_output.get("intent", "unknown")
                        route_reason = supervisor_output.get("reason", "")
                        route_detail = (
                            f"策略: {route_intent} | 来源: {route_source} | {route_reason}"
                        )
                    if grade:
                        grade_emoji = {
                            "highly_relevant": "✅",
                            "partially_relevant": "🔄",
                            "irrelevant": "⚠️",
                        }.get(grade, "⏳")
                        status_parts.append(f"{grade_emoji} 检索质量: {grade}")

                        if grade == "highly_relevant":
                            retrieval_detail = "✅ 检索质量高，文档直接包含答案"
                        elif grade == "partially_relevant":
                            retrieval_detail = "🔄 检索部分相关，已触发查询扩展补充"
                        else:
                            retrieval_detail = "⚠️ 检索相关性低，已启动查询改写"

                    if reflection > 0:
                        status_parts.append(f"🔄 Self-RAG反思迭代 #{reflection}")
                        retrieval_detail += f"\n正在进行第{reflection}轮反思检索..."

                    if eval_result:
                        support = eval_result.get("support_grade", "unknown")
                        if support == "fully_supported":
                            gen_detail = f"✅ 生成验证通过（支持度: {support}）"
                        elif support == "partially_supported":
                            gen_detail = "⚠️ 部分支持，已添加不确定性提示"
                        elif support == "no_support":
                            gen_detail = "🚨 检测到幻觉风险，已严格重生成"
                        else:
                            gen_detail = f"生成评估: {support}"

                    if state.get("final_answer"):
                        final = state["final_answer"]
                        if len(final) > len(full_response):
                            full_response = final
                            if history and history[-1]["role"] == "assistant":
                                history[-1]["content"] = full_response
                            else:
                                history.append(
                                    {"role": "assistant", "content": full_response}
                                )
                        status_parts.append("✅ 生成完成")

                    if citations:
                        cite_md = "**📚 引用来源：**\n"
                        for i, c in enumerate(citations[:3], 1):
                            source = c.get("source", "未知")
                            cite_grade = c.get("grade", "N/A")
                            score = c.get("score", "N/A")
                            if isinstance(score, float):
                                score = f"{score:.2f}"
                            badge_class = (
                                "grade-high"
                                if cite_grade == "highly_relevant"
                                else "grade-partial"
                                if cite_grade == "partially_relevant"
                                else "grade-low"
                            )
                            cite_md += f"{i}. `{source}` <span class='self-rag-badge {badge_class}'>{cite_grade}</span> (置信度: {score})\n"

                    debug_info = {}
                    if show_debug:
                        route_debug = {}
                        if supervisor_output:
                            route_debug = {
                                "intent": supervisor_output.get("intent"),
                                "route_source": supervisor_output.get("route_source"),
                                "route_reason": supervisor_output.get("reason"),
                                "route_confidence": supervisor_output.get("confidence"),
                                "should_try_search": supervisor_output.get("should_try_search"),
                                "should_try_retrieval": supervisor_output.get(
                                    "should_try_retrieval"
                                ),
                            }
                        debug_info = {
                            "routing": route_debug,
                            "retrieval_grade": grade,
                            "reflection_count": reflection,
                            "eval_result": eval_result,
                            "retrieval_count": state.get("retrieval_count"),
                            "next_step": state.get("next_step"),
                        }

                    current_status = (
                        " | ".join(status_parts) if status_parts else "处理中..."
                    )
                    yield (
                        history,
                        current_status if show_reasoning else "思考中...",
                        route_detail,
                        retrieval_detail,
                        gen_detail,
                        cite_md,
                        debug_info,
                        None,
                        {},
                        gr.update(visible=False),
                    )

                if uid and full_response:
                    memory_mgr = get_memory_manager(sess_id, uid)
                    memory_mgr.update_turn(current_question, full_response)
                    memory_mgr.finalize_session()

            except Exception as e:
                logger.error(f"[App] 处理失败: {e}")
                error_msg = f"❌ 错误: {str(e)}"
                history.append({"role": "assistant", "content": error_msg})
                yield (
                    history,
                    error_msg,
                    "路由失败",
                    "错误",
                    "错误",
                    "",
                    {},
                    None,
                    {},
                    gr.update(visible=False),
                )

        def apply_user_decision(
            history: List[dict],
            pending: dict,
            strategy_choice: str,
            low_conf_choice: str,
            worker_slot_input: str,
            thread_id_val: str,
            sess_id: str,
            uid: str,
            show_reasoning: bool,
            show_debug: bool,
        ) -> Generator:
            history = history or []
            if not pending:
                yield (
                    history,
                    "无待处理决策",
                    "等待路由...",
                    "等待查询...",
                    "等待生成...",
                    "",
                    {},
                    None,
                    {},
                    gr.update(visible=False),
                )
                return

            pending_type = pending.get("type")
            config_dict = {"configurable": {"thread_id": thread_id_val}}
            current_question = pending.get("question", "")

            if pending_type == "strategy_confirm":
                recommended = pending.get("recommended", "retrieval")
                chosen = strategy_choice if strategy_choice != "auto" else recommended
                history.append(
                    {
                        "role": "assistant",
                        "content": f"已确认策略: `{chosen}`，继续执行。",
                    }
                )
                yield from _stream_graph(
                    payload=Command(resume={"strategy": chosen}),
                    history=history,
                    config_dict=config_dict,
                    sess_id=sess_id,
                    uid=uid,
                    current_question=current_question,
                    show_reasoning=show_reasoning,
                    show_debug=show_debug,
                )
                return

            if pending_type == "low_confidence_answer":
                action_text = {
                    "accept": "已接受当前答案。",
                    "web_retry": "已选择补充联网重答。",
                    "conservative_retry": "已选择保守重答。",
                }.get(low_conf_choice, "已提交决策。")
                history.append({"role": "assistant", "content": action_text})
                yield from _stream_graph(
                    payload=Command(resume={"action": low_conf_choice}),
                    history=history,
                    config_dict=config_dict,
                    sess_id=sess_id,
                    uid=uid,
                    current_question=current_question,
                    show_reasoning=show_reasoning,
                    show_debug=show_debug,
                )
                return

            if pending_type == "shopping_slot_confirm":
                slot_answer = (worker_slot_input or "").strip()
                if not slot_answer:
                    slot_answer = "预算不限，优先性价比"
                history.append(
                    {
                        "role": "assistant",
                        "content": f"已补充导购偏好：`{slot_answer}`，继续检索。",
                    }
                )
                yield from _stream_graph(
                    payload=Command(resume={"slot_answer": slot_answer}),
                    history=history,
                    config_dict=config_dict,
                    sess_id=sess_id,
                    uid=uid,
                    current_question=current_question,
                    show_reasoning=show_reasoning,
                    show_debug=show_debug,
                )
                return

            yield (
                history,
                "未知决策类型",
                "等待路由...",
                "等待查询...",
                "等待生成...",
                "",
                {},
                None,
                {},
                gr.update(visible=False),
            )

        def show_memory(sess_id: str, uid: str):
            if not uid:
                return {"error": "未登录"}
            mgr = get_memory_manager(sess_id, uid)
            return {
                "short_term_summary": mgr.short_term.summary,
                "short_term_turns": len(mgr.short_term.messages) // 2,
                "long_term_facts": len(mgr.long_term.retrieve_relevant(uid, "test")),
            }

        def show_cache():
            return cache_manager.get_stats()

        def clear_current_memory(sess_id: str, uid: str):
            cleared = clear_memory_manager(sess_id, uid)
            return {
                "status": "已清除当前会话记忆" if cleared else "当前会话无内存缓存",
                "memory": {},
                "persistent": {},
            }

        submit_btn.click(
            process_message_stream,
            [
                msg_input,
                chatbot,
                thread_id,
                session_id,
                user_id,
                strategy_choice,
                show_reasoning,
                show_debug,
            ],
            [
                chatbot,
                status_text,
                route_text,
                retrieval_status,
                gen_status,
                citation_md,
                eval_json,
                interrupt_payload,
                hitl_json,
                decision_panel,
            ],
        )

        msg_input.submit(
            process_message_stream,
            [
                msg_input,
                chatbot,
                thread_id,
                session_id,
                user_id,
                strategy_choice,
                show_reasoning,
                show_debug,
            ],
            [
                chatbot,
                status_text,
                route_text,
                retrieval_status,
                gen_status,
                citation_md,
                eval_json,
                interrupt_payload,
                hitl_json,
                decision_panel,
            ],
        )

        decision_btn.click(
            apply_user_decision,
            [
                chatbot,
                interrupt_payload,
                strategy_choice,
                low_conf_choice,
                worker_slot_input,
                thread_id,
                session_id,
                user_id,
                show_reasoning,
                show_debug,
            ],
            [
                chatbot,
                status_text,
                route_text,
                retrieval_status,
                gen_status,
                citation_md,
                eval_json,
                interrupt_payload,
                hitl_json,
                decision_panel,
            ],
        )

        memory_btn.click(show_memory, [session_id, user_id], [memory_json])
        refresh_btn.click(show_cache, outputs=[cache_json])
        clear_memory_btn.click(
            clear_current_memory, [session_id, user_id], [memory_json]
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name=config.HOST,
        server_port=config.PORT,
        css=APP_CSS,
        show_error=True,
        quiet=False,
    )
