"""
responses.py - Standard JSON response envelopes for MachineMind AI.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
from flask import jsonify, Response


def get_utc_now_iso() -> str:
    """Returns current UTC timestamp formatted in ISO-8601 (e.g. 2026-01-31T12:00:00Z)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def success_response(data: Any = None, status_code: int = 200) -> Tuple[Response, int]:
    """Generates standard success response envelope."""
    payload = {
        "success": True,
        "data": data if data is not None else {},
        "error": None
    }
    return jsonify(payload), status_code


def error_response(
    code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    status_code: int = 400
) -> Tuple[Response, int]:
    """Generates standard error response envelope."""
    payload = {
        "success": False,
        "data": None,
        "error": {
            "code": code,
            "message": message,
            "details": details if details is not None else {}
        }
    }
    return jsonify(payload), status_code
