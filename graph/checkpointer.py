"""检查点管理"""

from typing import Dict


class CheckpointManager:
    """管理LangGraph检查点"""

    @staticmethod
    def get_state(thread_id: str) -> Dict:
        """获取状态快照"""
        from graph.builder import app

        config = {"configurable": {"thread_id": thread_id}}

        try:
            snapshot = app.get_state(config)
            return {
                "values": snapshot.values,
                "next": snapshot.next,
                "config": snapshot.config,
                "metadata": snapshot.metadata,
            }
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def get_state_history(thread_id: str) -> list:
        """获取执行历史"""
        from graph.builder import app

        config = {"configurable": {"thread_id": thread_id}}

        try:
            history = app.get_state_history(config)
            return [
                {
                    "values": h.values,
                    "next": h.next,
                    "timestamp": h.metadata.get("timestamp") if h.metadata else None,
                }
                for h in history
            ]
        except Exception as e:
            return [{"error": str(e)}]
