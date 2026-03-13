"""Helpers for consistent API responses."""

from typing import Any, Optional

from fastapi.responses import JSONResponse


class APIError(Exception):
    """Application-level API error with stable code/message."""

    def __init__(
        self,
        code: int,
        message: str,
        *,
        status_code: int = 400,
        data: Optional[Any] = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.data = data


def build_payload(
    request_id: str,
    *,
    code: int = 0,
    message: str = "ok",
    data: Optional[Any] = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "request_id": request_id,
        "data": data,
    }


def success_response(request_id: str, data: Optional[Any] = None) -> JSONResponse:
    return JSONResponse(build_payload(request_id, data=data))


def error_response(
    request_id: str,
    *,
    code: int,
    message: str,
    status_code: int,
    data: Optional[Any] = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=build_payload(
            request_id,
            code=code,
            message=message,
            data=data,
        ),
    )
