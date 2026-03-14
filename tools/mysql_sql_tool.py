"""MySQL 工具 - Few-shot 版本"""

from __future__ import annotations

import os
import re
from typing import Annotated
from urllib.parse import parse_qs, unquote, urlparse

import aiomysql
import pymysql
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from config import config
from llm_factory import get_llm
from prompts.registry import PromptRegistry

_schema_cache = None


def _row_value(row: dict, key: str):
    if key in row:
        return row[key]
    lowered = key.lower()
    uppered = key.upper()
    titled = key.title()
    for candidate in (lowered, uppered, titled):
        if candidate in row:
            return row[candidate]
    raise KeyError(key)


def get_db_url():
    return os.getenv(
        "DATABASE_URL", "mysql://root:123456@localhost:3306/enterprisemind"
    )


def _parse_mysql_url() -> dict:
    parsed = urlparse(get_db_url())
    if parsed.scheme not in {"mysql", "mysql+pymysql", "mysql+aiomysql"}:
        raise ValueError(f"仅支持 MySQL 数据库 URL，当前为: {parsed.scheme}")

    query = parse_qs(parsed.query)
    charset = query.get("charset", ["utf8mb4"])[0]
    return {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 3306,
        "user": unquote(parsed.username or "root"),
        "password": unquote(parsed.password or ""),
        "db": parsed.path.lstrip("/"),
        "charset": charset,
        "autocommit": True,
    }


def _connect_sync():
    params = _parse_mysql_url()
    return pymysql.connect(
        host=params["host"],
        port=params["port"],
        user=params["user"],
        password=params["password"],
        database=params["db"],
        charset=params["charset"],
        autocommit=params["autocommit"],
        cursorclass=pymysql.cursors.DictCursor,
    )


async def _connect_async():
    params = _parse_mysql_url()
    return await aiomysql.connect(
        host=params["host"],
        port=params["port"],
        user=params["user"],
        password=params["password"],
        db=params["db"],
        charset=params["charset"],
        autocommit=params["autocommit"],
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

    conn = _connect_sync()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = DATABASE()
                ORDER BY table_name, ordinal_position
                """
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    tables = {}
    for row in rows:
        table = _row_value(row, "table_name")
        if table not in tables:
            tables[table] = []
        column_name = _row_value(row, "column_name")
        data_type = _row_value(row, "data_type")
        tables[table].append(f"  {column_name} ({data_type})")

    parts = [f"\n表: {t}\n" + "\n".join(c) for t, c in tables.items()]
    _schema_cache = "\n".join(parts)
    return _schema_cache


async def get_schema_async() -> str:
    global _schema_cache
    if _schema_cache:
        return _schema_cache

    conn = await _connect_async()
    try:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                """
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = DATABASE()
                ORDER BY table_name, ordinal_position
                """
            )
            rows = await cur.fetchall()
    finally:
        conn.close()

    tables = {}
    for row in rows:
        table = _row_value(row, "table_name")
        if table not in tables:
            tables[table] = []
        column_name = _row_value(row, "column_name")
        data_type = _row_value(row, "data_type")
        tables[table].append(f"  {column_name} ({data_type})")

    parts = [f"\n表: {t}\n" + "\n".join(c) for t, c in tables.items()]
    _schema_cache = "\n".join(parts)
    return _schema_cache


def generate_sql_with_examples(question: str) -> str:
    """带 Few-shot 示例的 SQL 生成。"""
    schema = get_schema()
    llm = get_sql_llm()
    prompt_text = PromptRegistry.get(
        "sql_generator", variables={"schema": schema, "question": question}
    )

    response = llm.invoke(prompt_text)
    sql = response.content.strip()
    sql = re.sub(r"```sql|```", "", sql).strip()

    if not sql.strip().upper().startswith("SELECT"):
        raise ValueError("生成的SQL不是查询语句")

    if "LIMIT" not in sql.upper():
        sql = sql.rstrip(";") + " LIMIT 50;"

    return sql


async def generate_sql_with_examples_async(question: str) -> str:
    """带 Few-shot 示例的 SQL 生成（异步版）。"""
    schema = await get_schema_async()
    llm = get_sql_llm()
    prompt_text = PromptRegistry.get(
        "sql_generator", variables={"schema": schema, "question": question}
    )

    response = await llm.ainvoke(prompt_text)
    sql = response.content.strip()
    sql = re.sub(r"```sql|```", "", sql).strip()

    if not sql.strip().upper().startswith("SELECT"):
        raise ValueError("生成的SQL不是查询语句")

    if "LIMIT" not in sql.upper():
        sql = sql.rstrip(";") + " LIMIT 50;"

    return sql


def generate_sql(question: str) -> str:
    """向后兼容旧接口。"""
    return generate_sql_with_examples(question)


async def sql_query_async(question: str, safety_override: bool = False) -> dict:
    """自然语言转 SQL 并执行（异步版）。"""
    sql = await generate_sql_with_examples_async(question)

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
        return {"interrupt": "__INTERRUPT__", "sql": sql, "reason": "危险操作"}

    conn = await _connect_async()
    try:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(sql)
            rows = await cur.fetchall()
        data = [dict(row) for row in rows[:50]]
        llm = get_sql_llm()
        explain_prompt = f"""将以下数据总结为一句话回答用户问题。
数据：{str(data[:3])}
问题：{question}
回答："""
        explain = await llm.ainvoke(explain_prompt)
        return {
            "sql": sql,
            "summary": explain.content,
            "data": data[:10],
            "count": len(data),
        }
    except Exception as exc:
        return {"error": str(exc), "sql": sql}
    finally:
        conn.close()


async def sql_explain_async(question: str) -> str:
    return await generate_sql_with_examples_async(question)


@tool
def sql_query(
    question: Annotated[str, "自然语言查询问题"],
    safety_override: Annotated[bool, "跳过安全检查"] = False,
) -> str:
    """自然语言转 SQL 并执行。"""
    sql = generate_sql_with_examples(question)

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

    conn = _connect_sync()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
        data = [dict(row) for row in rows[:50]]

        llm = get_sql_llm()
        explain_prompt = f"""将以下数据总结为一句话回答用户问题。
数据：{str(data[:3])}
问题：{question}
回答："""
        explain = llm.invoke(explain_prompt)

        return {
            "sql": sql,
            "summary": explain.content,
            "data": data[:10],
            "count": len(data),
        }
    except Exception as exc:
        return {"error": str(exc), "sql": sql}
    finally:
        conn.close()


@tool
def sql_explain(question: Annotated[str, "查询问题"]) -> str:
    """仅生成 SQL，不执行。"""
    return generate_sql_with_examples(question)
