# state.py
from typing import TypedDict, List, Optional, Annotated, Literal, Dict, Any
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
import operator


class AgentState(TypedDict):
    """Agent状态 - Supervisor多Agent增强版"""

    messages: Annotated[List[AnyMessage], add_messages]

    question: str
    routing_hint: Optional[Literal["sql", "search", "calculation", "retrieval"]]
    retry_policy: Optional[Literal["accept", "web_retry", "conservative_retry"]]
    session_id: str
    user_id: Optional[str]
    hitl_request: Optional[Dict[str, Any]]

    # Self-RAG新增字段
    retrieval_grade: Optional[str]  # highly_relevant / partially_relevant / irrelevant
    self_rag_eval: Optional[Dict[str, Any]]  # 存储生成评估结果
    reflection_count: int  # 反思迭代计数器

    # 原有字段
    thought: Optional[str]
    action: Optional[str]
    action_input: Optional[str]
    observation: Optional[str]

    retrieved_docs: Annotated[List[dict], operator.add]
    retrieval_count: int

    tool_results: Annotated[List[dict], operator.add]

    generated_sql: Optional[str]
    sql_result: Optional[dict]

    # Supervisor协同状态
    active_agent: Optional[str]
    supervisor_decision: Optional[str]
    agent_outputs: Annotated[List[dict], operator.add]

    next_step: Literal[
        "supervisor",
        "sql_agent",
        "search_agent",
        "calculation_agent",
        "retrieval_agent",
        "response_agent",
        "hitl_strategy_confirm",
        "hitl_low_conf_confirm",
        "end",
    ]
    final_answer: Optional[str]
    citations: List[dict]

    working_memory: List[AnyMessage]
    token_usage: int
    execution_trace: Annotated[List[dict], operator.add]
