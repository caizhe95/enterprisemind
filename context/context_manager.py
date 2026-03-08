# context_manager.py
"""上下文管理 - 使用tiktoken精确计算"""

import tiktoken
from typing import List
from langchain_core.messages import SystemMessage, AIMessage, AnyMessage


class ContextManager:
    """Token预算管理（精确版）"""

    def __init__(self, max_tokens: int = 8000, model: str = "gpt-4"):
        self.max_tokens = max_tokens
        try:
            self.encoder = tiktoken.encoding_for_model(model)
        except Exception:
            self.encoder = tiktoken.get_encoding("cl100k_base")

    def estimate_tokens(self, text: str) -> int:
        """使用tiktoken精确计算token数"""
        if not text:
            return 0
        return len(self.encoder.encode(text))

    def build_messages(
        self,
        system_prompt: str,
        history: List[AnyMessage],
        retrieved_docs: List[dict],
        tool_results: List[dict],
    ) -> List[AnyMessage]:
        """构建消息（精确token控制）"""

        messages = []
        used_tokens = 0

        # System
        sys_tokens = self.estimate_tokens(system_prompt)
        if sys_tokens > 2000:
            system_prompt = system_prompt[:2000]  # 截断
            sys_tokens = self.estimate_tokens(system_prompt)

        sys_msg = SystemMessage(content=system_prompt)
        messages.append(sys_msg)
        used_tokens += sys_tokens

        # 预算分配
        remaining = self.max_tokens - used_tokens
        history_budget = int(remaining * 0.3)
        docs_budget = int(remaining * 0.6)

        # 历史（从最新开始）
        history_msgs = self._compress_history(history, history_budget)
        messages.extend(history_msgs)
        used_tokens += sum(self.estimate_tokens(m.content) for m in history_msgs)

        # 文档（动态截断）
        doc_msgs = self._format_docs_precise(retrieved_docs, docs_budget)
        messages.extend(doc_msgs)

        return messages

    def _compress_history(
        self, history: List[AnyMessage], budget: int
    ) -> List[AnyMessage]:
        """压缩历史（基于精确token数）"""
        if not history:
            return []

        result = []
        current_tokens = 0

        # 从后往前遍历（保留最近对话）
        for msg in reversed(history[-10:]):
            msg_tokens = self.estimate_tokens(msg.content)
            if current_tokens + msg_tokens > budget:
                break
            result.insert(0, msg)
            current_tokens += msg_tokens

        return result

    def _format_docs_precise(self, docs: List[dict], budget: int) -> List[AIMessage]:
        """精确格式化文档"""
        if not docs:
            return []

        messages = []
        used = 0

        for i, doc in enumerate(docs[:5]):
            content = doc.get("content", "")
            source = doc.get("metadata", {}).get("file_name", f"doc_{i}")

            # 动态截断内容以适应预算
            header = f"[来源: {source}]\n"
            header_tokens = self.estimate_tokens(header)
            remaining = budget - used - header_tokens - 20  # 20为缓冲

            if remaining <= 0:
                break

            # 截断内容到剩余token数（粗略估算：1token≈0.75字）
            max_chars = int(remaining * 0.75)
            truncated = content[:max_chars]

            text = header + truncated
            tokens = self.estimate_tokens(text)

            if used + tokens > budget:
                break

            messages.append(AIMessage(content=text))
            used += tokens

        return messages
