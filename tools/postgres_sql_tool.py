# postgres_sql_tool.py
"""PostgreSQL工具 - Few-shot版本"""

from typing import Annotated
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import psycopg2
from psycopg2.extras import RealDictCursor
import re
import os
from prompts.registry import PromptRegistry
from llm_factory import get_llm
from config import config

_schema_cache = None


def get_db_url():
    return os.getenv(
        "DATABASE_URL", "postgresql://postgres:123456@localhost:5432/enterprisemind"
    )


def get_sql_llm():
    """SQL 专用 LLM，优先复用项目配置，保留测试兼容入口。"""
    if config.RUN_MODE == "cloud":
        return get_llm()
    return ChatOpenAI(
        model=config.DEEPSEEK_MODEL,
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_API_BASE,
        temperature=0.3,
        max_tokens=4096,
    )


def get_schema():
    global _schema_cache
    if _schema_cache:
        return _schema_cache

    conn = psycopg2.connect(get_db_url())
    cur = conn.cursor()

    cur.execute("""
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
                ORDER BY table_name, ordinal_position
                """)

    tables = {}
    for row in cur.fetchall():
        table = row[0]
        if table not in tables:
            tables[table] = []
        tables[table].append(f"  {row[1]} ({row[2]})")

    parts = [f"\n表: {t}\n" + "\n".join(c) for t, c in tables.items()]
    _schema_cache = "\n".join(parts)
    conn.close()

    return _schema_cache


def generate_sql_with_examples(question: str) -> str:
    """带Few-shot示例的SQL生成"""
    schema = get_schema()
    llm = get_sql_llm()

    # 使用PromptRegistry获取带示例的模板
    prompt_text = PromptRegistry.get(
        "sql_generator", variables={"schema": schema, "question": question}
    )

    response = llm.invoke(prompt_text)
    sql = response.content.strip()
    sql = re.sub(r"```sql|```", "", sql).strip()

    # 强制安全：确保是SELECT
    if not sql.strip().upper().startswith("SELECT"):
        raise ValueError("生成的SQL不是查询语句")

    # 自动添加LIMIT（如果没有）
    if "LIMIT" not in sql.upper():
        sql = sql.rstrip(";") + " LIMIT 50;"

    return sql


def generate_sql(question: str) -> str:
    """向后兼容旧接口"""
    return generate_sql_with_examples(question)


@tool
def sql_query(
    question: Annotated[str, "自然语言查询问题"],
    safety_override: Annotated[bool, "跳过安全检查"] = False,
) -> str:
    """自然语言转SQL并执行"""

    sql = generate_sql_with_examples(question)

    # 多层安全检查
    dangerous = [
        "DELETE",
        "UPDATE",
        "INSERT",
        "DROP",
        "ALTER",
        "TRUNCATE",
        "GRANT",
        "REVOKE",
    ]
    is_dangerous = any(re.search(rf"\b{d}\b", sql, re.IGNORECASE) for d in dangerous)

    if is_dangerous and not safety_override:
        return f"__INTERRUPT__|sql_safety|{sql}|危险操作"

    # 执行查询
    conn = psycopg2.connect(get_db_url(), cursor_factory=RealDictCursor)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        data = [dict(row) for row in rows[:50]]

        # 生成自然语言解释
        llm = get_sql_llm()
        explain_prompt = f"""将以下数据总结为一句话回答用户问题。
数据：{str(data[:3])}
问题：{question}
回答："""
        explain = llm.invoke(explain_prompt)

        return {
            "sql": sql,
            "summary": explain.content,
            "data": data[:10],  # 限制返回数量
            "count": len(data),
        }
    except Exception as e:
        return {"error": str(e), "sql": sql}
    finally:
        conn.close()


@tool
def sql_explain(question: Annotated[str, "查询问题"]) -> str:
    """仅生成SQL，不执行"""
    return generate_sql_with_examples(question)
