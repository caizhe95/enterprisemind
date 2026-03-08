"""三层记忆管理系统"""

from typing import List, Dict, Optional
from datetime import datetime
import json
import sqlite3
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from llm_factory import get_llm
from logger import logger


class PerceptualMemory:
    """感知记忆：当前会话原始对话"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: List[BaseMessage] = []

    def add_turn(self, human_msg: str, ai_msg: str):
        self.messages.extend(
            [HumanMessage(content=human_msg), AIMessage(content=ai_msg)]
        )

    def get_recent(self, n: int = 3) -> List[BaseMessage]:
        """获取最近n轮"""
        return self.messages[-n * 2 :] if len(self.messages) > n * 2 else self.messages


class WorkingMemory:
    """工作记忆：压缩后的上下文（解决Token爆炸）"""

    def __init__(self, max_tokens: int = 2000):
        self.max_tokens = max_tokens
        self.compress_threshold = 6  # 6轮后触发压缩
        self.turn_count = 0
        self.summary = ""  # 累积摘要
        self.recent_raw: List[BaseMessage] = []  # 未压缩的近期

    def add_message(self, msg: BaseMessage):
        self.recent_raw.append(msg)
        self.turn_count += 0.5  # Human和AI各算0.5轮

        # 首次达到6轮触发；已有摘要后按更短窗口滚动压缩，避免再次堆积
        threshold = 2 if self.summary else self.compress_threshold
        if self.turn_count >= threshold:
            self._compress()

    def _compress(self):
        """LLM压缩历史"""
        if len(self.recent_raw) < 4:
            return

        to_compress = self.recent_raw[:-2]  # 保留最近2轮

        conv_text = "\n".join(
            [
                f"User: {m.content}"
                if isinstance(m, HumanMessage)
                else f"AI: {m.content}"
                for m in to_compress
            ]
        )

        prompt = f"""将以下对话压缩为关键事实摘要（保留：用户身份、偏好、业务关键数据），50字以内：

{conv_text}

现有摘要：{self.summary}
新的累积摘要："""

        try:
            llm = get_llm()
            new_summary = llm.invoke(prompt).content.strip()
            self.summary = new_summary
            self.recent_raw = self.recent_raw[-2:]  # 只保留最近2轮
            self.turn_count = 0
            logger.info(f"[WorkingMemory] 压缩完成: {self.summary[:50]}...")
        except Exception as e:
            logger.error(f"[WorkingMemory] 压缩失败: {e}")
            # 离线回退：失败时仍完成压缩，保证会话可持续
            fallback = conv_text.replace("\n", " ")[:80]
            self.summary = self.summary or f"对话摘要: {fallback}"
            self.recent_raw = self.recent_raw[-2:]
            self.turn_count = 0

    def get_context(self) -> str:
        """构建上下文"""
        parts = []
        if self.summary:
            parts.append(f"[历史摘要]: {self.summary}")

        if self.recent_raw:
            parts.append("[最近对话]:")
            parts.extend(
                [
                    f"User: {m.content}"
                    if isinstance(m, HumanMessage)
                    else f"AI: {m.content[:200]}"
                    for m in self.recent_raw
                ]
            )

        return "\n".join(parts)


class LongTermMemory:
    """长期记忆：跨会话持久化"""

    def __init__(self, db_path: str = "./memory/long_term.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._init_db()
        except Exception as e:
            logger.error(f"[LongTermMemory] 初始化失败，切换本地路径: {e}")
            fallback_name = (
                f"long_term_fallback_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.db"
            )
            self.db_path = Path("./memory") / fallback_name
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

        try:
            self.llm = get_llm()
        except Exception as e:
            logger.warning(f"[LongTermMemory] LLM不可用，启用规则提取: {e}")
            self.llm = None

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            # 用户事实表
            conn.execute("""
                         CREATE TABLE IF NOT EXISTS user_facts
                         (
                             id         INTEGER PRIMARY KEY,
                             user_id    TEXT NOT NULL,
                             fact_type  TEXT NOT NULL, -- 'preference', 'profile', 'business'
                             fact_key   TEXT NOT NULL,
                             fact_value TEXT NOT NULL,
                             confidence REAL      DEFAULT 1.0,
                             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                             updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                             UNIQUE (user_id, fact_type, fact_key)
                         )
                         """)

            # 会话摘要表
            conn.execute("""
                         CREATE TABLE IF NOT EXISTS session_summaries
                         (
                             session_id TEXT PRIMARY KEY,
                             user_id    TEXT NOT NULL,
                             summary    TEXT NOT NULL,
                             topics     TEXT, -- JSON数组
                             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                         )
                         """)
            conn.commit()

    def extract_facts(self, user_id: str, messages: List[BaseMessage]) -> List[Dict]:
        """从对话中提取结构化事实"""
        if not messages or not user_id:
            return []

        recent = messages[-6:]  # 最近3轮
        conv_text = "\n".join(
            [
                f"User: {m.content}"
                if isinstance(m, HumanMessage)
                else f"AI: {m.content[:100]}"
                for m in recent
            ]
        )

        prompt = f"""从对话中提取用户明确陈述的事实，返回JSON数组：
事实类型：preference(偏好), profile(身份), business(业务规则)

对话：
{conv_text}

提取规则：
1. 只提取"我喜欢/我是/我需要/我们单位"等明确陈述
2. 忽略临时性问题（如"现在几点"）
3. confidence: 0-1之间的置信度

格式：[{{"type": "preference", "key": "分析工具", "value": "Excel", "confidence": 0.9}}]

结果："""

        try:
            if self.llm is not None:
                response = self.llm.invoke(prompt).content
                import re

                match = re.search(r"\[.*\]", response, re.DOTALL)
                if match:
                    facts = json.loads(match.group())
                    parsed = [
                        f
                        for f in facts
                        if isinstance(f, dict) and f.get("confidence", 0) > 0.7
                    ]
                    if parsed:
                        return parsed
        except Exception as e:
            logger.error(f"[LongTermMemory] 提取失败: {e}")

        # 规则回退：保障离线/弱网环境也能提取基础事实
        extracted: List[Dict] = []
        for m in recent:
            if not isinstance(m, HumanMessage):
                continue
            content = str(m.content)
            if "我喜欢" in content or "偏好" in content:
                value = (
                    content.split("我喜欢", 1)[-1].strip()
                    if "我喜欢" in content
                    else content
                )
                extracted.append(
                    {
                        "type": "preference",
                        "key": "用户偏好",
                        "value": value,
                        "confidence": 0.85,
                    }
                )
            if "我是" in content:
                value = content.split("我是", 1)[-1].strip()
                extracted.append(
                    {
                        "type": "profile",
                        "key": "用户身份",
                        "value": value,
                        "confidence": 0.85,
                    }
                )
            if "我们" in content or "公司" in content or "单位" in content:
                extracted.append(
                    {
                        "type": "business",
                        "key": "业务信息",
                        "value": content[:80],
                        "confidence": 0.8,
                    }
                )

        return extracted

    def save_facts(self, user_id: str, facts: List[Dict]):
        """保存事实"""
        if not user_id or not facts:
            return

        with sqlite3.connect(self.db_path) as conn:
            for fact in facts:
                conn.execute(
                    """
                             INSERT INTO user_facts (user_id, fact_type, fact_key, fact_value, confidence)
                             VALUES (?, ?, ?, ?, ?)
                             ON CONFLICT(user_id, fact_type, fact_key)
                                 DO UPDATE SET fact_value=excluded.fact_value,
                                               confidence=excluded.confidence,
                                               updated_at=CURRENT_TIMESTAMP
                             """,
                    (
                        user_id,
                        fact.get("type", "general"),
                        fact.get("key"),
                        fact.get("value"),
                        fact.get("confidence", 1.0),
                    ),
                )
            conn.commit()

        logger.info(f"[LongTermMemory] 保存 {len(facts)} 条事实 for {user_id}")

    def retrieve_relevant(self, user_id: str, query: str, top_k: int = 3) -> List[str]:
        """检索相关记忆"""
        if not user_id:
            return []

        # 简单关键词匹配（可升级为向量检索）
        keywords = set(query.lower().split()) - set(["的", "了", "是", "什么", "怎么"])

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT fact_value, confidence FROM user_facts WHERE user_id = ?",
                (user_id,),
            )
            facts = cursor.fetchall()

        # 相关性打分
        scored = []
        for value, conf in facts:
            score = sum(1 for k in keywords if k in value.lower()) * conf
            if score > 0:
                scored.append((score, value))

        scored.sort(reverse=True)
        return [v for _, v in scored[:top_k]]

    def save_session_summary(
        self, session_id: str, user_id: str, summary: str, topics: List[str] = None
    ):
        """保存会话摘要"""
        if not session_id or not user_id:
            return

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO session_summaries 
                (session_id, user_id, summary, topics)
                VALUES (?, ?, ?, ?)
            """,
                (session_id, user_id, summary, json.dumps(topics or [])),
            )
            conn.commit()


class MemoryManager:
    """记忆管理器门面"""

    def __init__(self, session_id: str, user_id: Optional[str] = None):
        self.session_id = session_id
        self.user_id = user_id

        self.perceptual = PerceptualMemory(session_id)
        self.working = WorkingMemory()
        self.long_term = LongTermMemory()

        if user_id:
            self._load_user_context()

    def _load_user_context(self):
        """加载用户历史"""
        # 获取最近会话摘要
        with sqlite3.connect(self.long_term.db_path) as conn:
            cursor = conn.execute(
                """SELECT summary
                   FROM session_summaries
                   WHERE user_id = ?
                   ORDER BY created_at DESC
                   LIMIT 1""",
                (self.user_id,),
            )
            row = cursor.fetchone()
            if row:
                self.working.summary = f"[上次对话: {row[0]}]"
                logger.info(f"[MemoryManager] 已加载用户 {self.user_id} 的历史")

    def get_context_for_query(self, query: str) -> str:
        """为查询构建记忆上下文"""
        parts = []

        # 工作记忆
        working_ctx = self.working.get_context()
        if working_ctx:
            parts.append(working_ctx)

        # 相关长期记忆
        if self.user_id:
            relevant = self.long_term.retrieve_relevant(self.user_id, query)
            if relevant:
                parts.append("[用户偏好]:\n- " + "\n- ".join(relevant))

        context = "\n\n".join(parts)
        return context[:2000]  # 硬限制防止Token爆炸

    def update_turn(self, user_msg: str, ai_msg: str):
        """更新记忆（每轮调用）"""
        # 感知记忆
        self.perceptual.add_turn(user_msg, ai_msg)

        # 工作记忆
        self.working.add_message(HumanMessage(content=user_msg))
        self.working.add_message(AIMessage(content=ai_msg))

        # 长期记忆（每4轮提取一次，避免频繁调用LLM）
        if len(self.perceptual.messages) % 4 == 0 and self.user_id:
            facts = self.long_term.extract_facts(self.user_id, self.perceptual.messages)
            if facts:
                self.long_term.save_facts(self.user_id, facts)

    def finalize_session(self):
        """会话结束保存"""
        if not self.user_id:
            return

        summary = self.working.summary or "无摘要"
        self.long_term.save_session_summary(self.session_id, self.user_id, summary)
        logger.info(f"[MemoryManager] 会话 {self.session_id} 已保存")


# 全局管理器缓存（简化版）
_memory_managers: Dict[str, MemoryManager] = {}


def get_memory_manager(session_id: str, user_id: Optional[str] = None) -> MemoryManager:
    """获取记忆管理器"""
    key = f"{user_id or 'anon'}:{session_id}"
    if key not in _memory_managers:
        _memory_managers[key] = MemoryManager(session_id, user_id)
    return _memory_managers[key]


def clear_memory_manager(session_id: str, user_id: Optional[str] = None) -> bool:
    """清除当前会话记忆管理器缓存。"""
    key = f"{user_id or 'anon'}:{session_id}"
    existed = key in _memory_managers
    _memory_managers.pop(key, None)
    return existed
