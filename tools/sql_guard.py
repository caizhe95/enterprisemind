"""SQL guard tool for read-only enforcement."""

from __future__ import annotations

import re
from typing import Annotated, Any

from langchain_core.tools import tool


@tool
def sql_guard(sql: Annotated[str, "待检查的SQL"]) -> dict[str, Any]:
    """Check whether SQL is safe to execute in read-only mode."""

    dangerous = ["DELETE", "UPDATE", "INSERT", "DROP", "ALTER", "TRUNCATE", "GRANT", "REVOKE"]
    hits = [item for item in dangerous if re.search(rf"\b{item}\b", sql or "", re.IGNORECASE)]
    if hits:
        return {
            "allowed": False,
            "reason": f"检测到危险SQL关键字: {', '.join(hits)}",
            "matched_keywords": hits,
        }
    return {"allowed": True, "reason": "SQL安全检查通过", "matched_keywords": []}
