"""API response schemas."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class APIResponse(BaseModel):
    code: int = Field(default=0)
    message: str = Field(default="ok")
    request_id: str
    data: Optional[Any] = None
