"""FastAPI service for shopping multi-agent assistant."""

import time
import uuid
from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langgraph.types import Command

from config import check_environment
from graph.builder import app as graph_app
from graph.state_helpers import build_initial_state, normalize_interrupt
from memory.memory_manager import get_memory_manager

check_environment()

api = FastAPI(
    title="智能导购多 Agent 系统 API",
    description="多 Worker 智能导购 API，支持检索、结构化抽取、推荐、计算、SQL 分析与外部搜索。",
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    thread_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    routing_hint: Literal["sql", "search", "calculation", "retrieval", "auto"] = "auto"


class DecisionRequest(BaseModel):
    thread_id: str
    action: Optional[Literal["accept", "web_retry", "conservative_retry"]] = None
    strategy: Optional[
        Literal["auto", "sql", "search", "calculation", "retrieval"]
    ] = None
    slot_answer: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None

def _run_graph(payload: Any, thread_id: str) -> Dict[str, Any]:
    config_dict = {"configurable": {"thread_id": thread_id}}
    final_state = None
    pending = None

    for event in graph_app.stream(payload, config_dict, stream_mode="values"):
        if isinstance(event, dict) and "__interrupt__" in event:
            pending = normalize_interrupt(event["__interrupt__"])
            break
        final_state = event

    return {"final_state": final_state, "pending": pending}


def _finalize_memory(
    state: Optional[Dict[str, Any]],
    fallback_session: str,
    fallback_user: Optional[str],
):
    if not state:
        return
    answer = state.get("final_answer")
    question = state.get("question")
    user_id = state.get("user_id") or fallback_user
    session_id = state.get("session_id") or fallback_session
    if user_id and question and answer:
        memory_mgr = get_memory_manager(session_id, user_id)
        memory_mgr.update_turn(question, answer)
        memory_mgr.finalize_session()


@api.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@api.post("/chat")
def chat(req: ChatRequest) -> Dict[str, Any]:
    thread_id = req.thread_id or str(uuid.uuid4())
    session_id = req.session_id or f"sess_{int(time.time())}"
    user_id = req.user_id or f"user_{int(time.time())}"
    routing_hint = None if req.routing_hint == "auto" else req.routing_hint

    state = build_initial_state(req.question, session_id, user_id, routing_hint)
    result = _run_graph(state, thread_id)

    if result["pending"]:
        return {
            "thread_id": thread_id,
            "session_id": session_id,
            "user_id": user_id,
            "status": "interrupt",
            "pending": result["pending"],
        }

    final_state = result["final_state"] or {}
    _finalize_memory(final_state, session_id, user_id)
    return {
        "thread_id": thread_id,
        "session_id": session_id,
        "user_id": user_id,
        "status": "completed",
        "answer": final_state.get("final_answer"),
        "citations": final_state.get("citations", []),
        "retrieval_grade": final_state.get("retrieval_grade"),
        "self_rag_eval": final_state.get("self_rag_eval"),
    }


@api.post("/decision")
def decision(req: DecisionRequest) -> Dict[str, Any]:
    if not req.action and not req.strategy and not req.slot_answer:
        raise HTTPException(
            status_code=400, detail="action、strategy、slot_answer 至少提供一个"
        )

    resume_payload = {}
    if req.strategy:
        resume_payload["strategy"] = req.strategy
    if req.action:
        resume_payload["action"] = req.action
    if req.slot_answer:
        resume_payload["slot_answer"] = req.slot_answer

    result = _run_graph(Command(resume=resume_payload), req.thread_id)

    if result["pending"]:
        return {
            "thread_id": req.thread_id,
            "session_id": req.session_id,
            "user_id": req.user_id,
            "status": "interrupt",
            "pending": result["pending"],
        }

    final_state = result["final_state"] or {}
    _finalize_memory(final_state, req.session_id or "", req.user_id)
    return {
        "thread_id": req.thread_id,
        "session_id": final_state.get("session_id", req.session_id),
        "user_id": final_state.get("user_id", req.user_id),
        "status": "completed",
        "answer": final_state.get("final_answer"),
        "citations": final_state.get("citations", []),
        "retrieval_grade": final_state.get("retrieval_grade"),
        "self_rag_eval": final_state.get("self_rag_eval"),
    }
