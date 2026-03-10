from pathlib import Path
import uuid

from memory.memory_manager import MemoryManager, ShortTermMemory, LongTermMemory


class _DummyResponse:
    def __init__(self, content: str):
        self.content = content


class _DummyLLM:
    def invoke(self, prompt: str):
        return _DummyResponse("用户偏好轻薄续航")


def test_short_term_memory_compresses_history(monkeypatch):
    monkeypatch.setattr("memory.memory_manager.get_llm", lambda: _DummyLLM())

    memory = ShortTermMemory()
    for idx in range(6):
        memory.add_turn(f"user-{idx}", f"ai-{idx}")

    assert memory.summary == "用户偏好轻薄续航"
    assert len(memory.recent_raw) <= 2
    assert len(memory.messages) == 12


def test_memory_manager_uses_short_and_long_term():
    db_path = Path("tests") / f"memory_test_{uuid.uuid4().hex}.db"
    manager = MemoryManager("sess_1", "user_1")
    manager.long_term = LongTermMemory(str(db_path))

    manager.short_term.summary = "预算5000，偏好轻薄"
    manager.long_term.save_facts(
        "user_1",
        [
            {
                "type": "preference",
                "key": "用户偏好",
                "value": "偏好轻薄笔记本",
                "confidence": 0.9,
            }
        ],
    )

    context = manager.get_context_for_query("偏好轻薄笔记本")

    assert "预算5000" in context
    assert "偏好轻薄笔记本" in context
    assert manager.working is manager.short_term
    assert manager.perceptual is manager.short_term
