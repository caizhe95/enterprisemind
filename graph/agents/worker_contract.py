"""Normalized worker contract for shopping-domain orchestration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_worker_output(
    *,
    worker: str,
    status: str,
    summary: str,
    artifacts: Optional[Dict[str, Any]] = None,
    signals: Optional[List[str]] = None,
    confidence: float = 0.5,
    errors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "worker": worker,
        "status": status,
        "summary": summary,
        "artifacts": artifacts or {},
        "signals": signals or [],
        "confidence": confidence,
        "errors": errors or [],
    }
