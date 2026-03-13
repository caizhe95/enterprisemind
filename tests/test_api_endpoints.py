from fastapi.testclient import TestClient

import server


client = TestClient(server.api)


def test_health_returns_standard_response():
    response = client.get("/health", headers={"x-request-id": "req-health-001"})

    body = response.json()
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req-health-001"
    assert body["code"] == 0
    assert body["message"] == "ok"
    assert body["request_id"] == "req-health-001"
    assert body["data"]["status"] == "ok"


def test_chat_returns_standard_response_for_completed_flow(monkeypatch):
    async def fake_run_graph_async(payload, thread_id):
        return {
            "pending": None,
            "final_state": {
                "final_answer": "测试答案",
                "citations": [{"source": "products.md"}],
                "retrieval_grade": "highly_relevant",
                "self_rag_eval": {"support_grade": "fully_supported"},
                "agent_outputs": [
                    {"agent": "supervisor"},
                    {"agent": "planner"},
                    {"agent": "response_agent"},
                ],
            },
        }

    async def fake_finalize_memory_async(*args, **kwargs):
        return None

    monkeypatch.setattr(server, "_run_graph_async", fake_run_graph_async)
    monkeypatch.setattr(server, "_finalize_memory_async", fake_finalize_memory_async)

    response = client.post(
        "/chat",
        json={"question": "星澜手机1代的价格是多少？"},
        headers={"x-request-id": "req-chat-001"},
    )

    body = response.json()
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req-chat-001"
    assert body["code"] == 0
    assert body["message"] == "ok"
    assert body["request_id"] == "req-chat-001"
    assert body["data"]["status"] == "completed"
    assert body["data"]["answer"] == "测试答案"
    assert body["data"]["agent_path"] == ["supervisor", "planner", "response_agent"]


def test_chat_returns_standard_response_for_interrupt(monkeypatch):
    async def fake_run_graph_async(payload, thread_id):
        return {
            "pending": {"type": "strategy_confirm", "question": "测试问题"},
            "final_state": None,
        }

    monkeypatch.setattr(server, "_run_graph_async", fake_run_graph_async)

    response = client.post("/chat", json={"question": "帮我推荐一个手机"})

    body = response.json()
    assert response.status_code == 200
    assert body["code"] == 0
    assert body["message"] == "ok"
    assert body["request_id"]
    assert body["data"]["status"] == "interrupt"
    assert body["data"]["pending"]["type"] == "strategy_confirm"


def test_decision_requires_at_least_one_resume_field():
    response = client.post("/decision", json={"thread_id": "thread-1"})

    body = response.json()
    assert response.status_code == 400
    assert body["code"] == 4001
    assert body["message"] == "action、strategy、slot_answer 至少提供一个"
