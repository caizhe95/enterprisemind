"""FastAPI service for shopping multi-agent assistant."""

import time
import uuid
from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from langgraph.types import Command
from starlette.concurrency import run_in_threadpool

from config import check_environment
from graph.builder import app as graph_app
from graph.state_helpers import build_initial_state, normalize_interrupt
from logger import logger
from memory.memory_manager import get_memory_manager
from schemas.api import APIResponse
from utils.response import APIError, error_response, success_response

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


def _get_request_id(request: Request) -> str:
    return request.headers.get("x-request-id") or str(uuid.uuid4())


def _collect_agent_path(state: Optional[Dict[str, Any]]) -> list[str]:
    if not state:
        return []
    outputs = state.get("agent_outputs", []) or []
    return [item["agent"] for item in outputs if item.get("agent")]

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


async def _run_graph_async(payload: Any, thread_id: str) -> Dict[str, Any]:
    return await run_in_threadpool(_run_graph, payload, thread_id)


async def _finalize_memory_async(
    state: Optional[Dict[str, Any]],
    fallback_session: str,
    fallback_user: Optional[str],
):
    await run_in_threadpool(_finalize_memory, state, fallback_session, fallback_user)


@api.exception_handler(APIError)
def handle_api_error(request: Request, exc: APIError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.warning(
        "api_error request_id={} path={} code={} message={}",
        request_id,
        request.url.path,
        exc.code,
        exc.message,
    )
    return error_response(
        request_id,
        code=exc.code,
        message=exc.message,
        status_code=exc.status_code,
        data=exc.data,
    )


@api.exception_handler(HTTPException)
def handle_http_error(request: Request, exc: HTTPException) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return error_response(
        request_id,
        code=exc.status_code,
        message=str(exc.detail),
        status_code=exc.status_code,
    )


@api.exception_handler(RequestValidationError)
def handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return error_response(
        request_id,
        code=4000,
        message="invalid request",
        status_code=422,
        data={"errors": exc.errors()},
    )


@api.exception_handler(Exception)
def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.exception(
        "unhandled_error request_id={} path={} error={}",
        request_id,
        request.url.path,
        str(exc),
    )
    return error_response(
        request_id,
        code=5000,
        message="internal server error",
        status_code=500,
    )


@api.middleware("http")
async def attach_request_context(request: Request, call_next):
    request_id = _get_request_id(request)
    request.state.request_id = request_id
    started_at = time.perf_counter()
    logger.info("request_started request_id={} path={}", request_id, request.url.path)
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_finished request_id={} path={} status_code={} elapsed_ms={}",
        request_id,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@api.get("/health", response_model=APIResponse)
async def health(request: Request) -> JSONResponse:
    return success_response(request.state.request_id, {"status": "ok"})


@api.get("/ready", response_model=APIResponse)
async def ready(request: Request) -> JSONResponse:
    runtime_config = await run_in_threadpool(check_environment)
    return success_response(
        request.state.request_id,
        {
            "status": "ready",
            "run_mode": runtime_config.RUN_MODE,
        },
    )


@api.post("/chat", response_model=APIResponse)
async def chat(request: Request, req: ChatRequest) -> JSONResponse:
    request_id = request.state.request_id
    started_at = time.perf_counter()
    thread_id = req.thread_id or str(uuid.uuid4())
    session_id = req.session_id or f"sess_{int(time.time())}"
    user_id = req.user_id or f"user_{int(time.time())}"
    routing_hint = None if req.routing_hint == "auto" else req.routing_hint

    logger.info(
        "chat_started request_id={} thread_id={} session_id={} user_id={} routing_hint={} question={}",
        request_id,
        thread_id,
        session_id,
        user_id,
        req.routing_hint,
        req.question,
    )

    state = build_initial_state(req.question, session_id, user_id, routing_hint)
    result = await _run_graph_async(state, thread_id)

    if result["pending"]:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        logger.info(
            "chat_interrupted request_id={} thread_id={} elapsed_ms={} pending_type={}",
            request_id,
            thread_id,
            elapsed_ms,
            result["pending"].get("type"),
        )
        return success_response(
            request_id,
            {
                "thread_id": thread_id,
                "session_id": session_id,
                "user_id": user_id,
                "status": "interrupt",
                "pending": result["pending"],
            },
        )

    final_state = result["final_state"] or {}
    await _finalize_memory_async(final_state, session_id, user_id)
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
    logger.info(
        "chat_completed request_id={} thread_id={} elapsed_ms={} agent_path={}",
        request_id,
        thread_id,
        elapsed_ms,
        _collect_agent_path(final_state),
    )
    return success_response(
        request_id,
        {
            "thread_id": thread_id,
            "session_id": session_id,
            "user_id": user_id,
            "status": "completed",
            "answer": final_state.get("final_answer"),
            "citations": final_state.get("citations", []),
            "retrieval_grade": final_state.get("retrieval_grade"),
            "self_rag_eval": final_state.get("self_rag_eval"),
            "agent_path": _collect_agent_path(final_state),
        },
    )


@api.post("/decision", response_model=APIResponse)
async def decision(request: Request, req: DecisionRequest) -> JSONResponse:
    request_id = request.state.request_id
    started_at = time.perf_counter()
    if not req.action and not req.strategy and not req.slot_answer:
        raise APIError(
            4001,
            "action、strategy、slot_answer 至少提供一个",
            status_code=400,
        )

    resume_payload = {}
    if req.strategy:
        resume_payload["strategy"] = req.strategy
    if req.action:
        resume_payload["action"] = req.action
    if req.slot_answer:
        resume_payload["slot_answer"] = req.slot_answer

    logger.info(
        "decision_started request_id={} thread_id={} action={} strategy={} has_slot_answer={}",
        request_id,
        req.thread_id,
        req.action,
        req.strategy,
        bool(req.slot_answer),
    )

    result = await _run_graph_async(Command(resume=resume_payload), req.thread_id)

    if result["pending"]:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        logger.info(
            "decision_interrupted request_id={} thread_id={} elapsed_ms={} pending_type={}",
            request_id,
            req.thread_id,
            elapsed_ms,
            result["pending"].get("type"),
        )
        return success_response(
            request_id,
            {
                "thread_id": req.thread_id,
                "session_id": req.session_id,
                "user_id": req.user_id,
                "status": "interrupt",
                "pending": result["pending"],
            },
        )

    final_state = result["final_state"] or {}
    await _finalize_memory_async(final_state, req.session_id or "", req.user_id)
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
    logger.info(
        "decision_completed request_id={} thread_id={} elapsed_ms={} agent_path={}",
        request_id,
        req.thread_id,
        elapsed_ms,
        _collect_agent_path(final_state),
    )
    return success_response(
        request_id,
        {
            "thread_id": req.thread_id,
            "session_id": final_state.get("session_id", req.session_id),
            "user_id": final_state.get("user_id", req.user_id),
            "status": "completed",
            "answer": final_state.get("final_answer"),
            "citations": final_state.get("citations", []),
            "retrieval_grade": final_state.get("retrieval_grade"),
            "self_rag_eval": final_state.get("self_rag_eval"),
            "agent_path": _collect_agent_path(final_state),
        },
    )
