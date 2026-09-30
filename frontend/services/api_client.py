"""
api_client.py - Centralized HTTP client for talking to MachineMind AI Flask Backend.

ONLY module in the frontend that performs HTTP REST calls using 'requests'.
Handles connection failures, HTTP timeouts, and API error envelopes.
"""

import logging
from typing import Any, Dict, List, Optional
import requests
import streamlit as st

from frontend.utils.config import BACKEND_URL, READ_TIMEOUT, INFERENCE_TIMEOUT

logger = logging.getLogger(__name__)


class ApiError(Exception):
    """Custom exception raised when backend returns an error or is unreachable."""

    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def _request(
    method: str,
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
    json_data: Optional[Dict[str, Any]] = None,
    files: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    timeout: int = READ_TIMEOUT
) -> Dict[str, Any]:
    """Helper method executing HTTP request and parsing standard backend JSON envelope."""
    url = f"{BACKEND_URL}/api/{endpoint.lstrip('/')}"

    try:
        response = requests.request(
            method=method,
            url=url,
            params=params,
            json=json_data,
            files=files,
            data=data,
            timeout=timeout
        )
    except requests.exceptions.ConnectionError:
        raise ApiError(
            code="CONNECTION_ERROR",
            message=f"Backend server is unreachable at {BACKEND_URL}. Please ensure Flask backend is running.",
            details={"url": url}
        )
    except requests.exceptions.Timeout:
        raise ApiError(
            code="TIMEOUT_ERROR",
            message=f"Backend request to {endpoint} timed out after {timeout} seconds.",
            details={"url": url, "timeout": timeout}
        )
    except Exception as e:
        raise ApiError(
            code="CLIENT_ERROR",
            message=f"HTTP request failed: {str(e)}",
            details={"url": url, "error": str(e)}
        )

    # Attempt parsing JSON response envelope
    try:
        payload = response.json()
    except Exception:
        raise ApiError(
            code="INVALID_RESPONSE",
            message=f"Backend returned non-JSON response (HTTP {response.status_code}).",
            details={"status_code": response.status_code, "text": response.text[:200]}
        )

    if not isinstance(payload, dict):
        raise ApiError(
            code="INVALID_RESPONSE",
            message="Malformed response payload from backend.",
            details={"payload": str(payload)[:200]}
        )

    # Check backend envelope { success: bool, data: ..., error: ... }
    success = payload.get("success", False)
    if success and response.status_code in (200, 201):
        return payload.get("data")

    # Extract error payload
    err_dict = payload.get("error") or {}
    code = err_dict.get("code", f"HTTP_{response.status_code}")
    message = err_dict.get("message", f"Backend request failed with status code {response.status_code}.")
    details = err_dict.get("details", {})

    raise ApiError(code=code, message=message, details=details)


@st.cache_data(ttl=5)
def get_health() -> Dict[str, Any]:
    """GET /api/health"""
    return _request("GET", "health")


@st.cache_data(ttl=5)
def list_machines() -> List[Dict[str, Any]]:
    """GET /api/machines"""
    res = _request("GET", "machines")
    return res.get("machines", [])


@st.cache_data(ttl=5)
def get_machine(machine_id: str) -> Dict[str, Any]:
    """GET /api/machines/<machine_id>"""
    return _request("GET", f"machines/{machine_id}")


def create_machine(machine_id: str, name: str, description: Optional[str] = None) -> Dict[str, Any]:
    """POST /api/machines"""
    payload = {"machine_id": machine_id, "name": name, "description": description or ""}
    res = _request("POST", "machines", json_data=payload)
    st.cache_data.clear()
    return res


def run_inference(
    machine_id: str,
    file_bytes: bytes,
    filename: str,
    snapshot_time: Optional[str] = None
) -> Dict[str, Any]:
    """
    POST /api/inference (multipart form-data upload)
    NEVER CACHED.
    """
    files = {"file": (filename, file_bytes)}
    data = {"machine_id": machine_id}
    if snapshot_time:
        data["snapshot_time"] = snapshot_time

    res = _request(
        "POST",
        "inference",
        files=files,
        data=data,
        timeout=INFERENCE_TIMEOUT
    )
    st.cache_data.clear()
    return res


@st.cache_data(ttl=5)
def get_predictions(machine_id: str, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    """GET /api/predictions/<machine_id>"""
    return _request("GET", f"predictions/{machine_id}", params={"limit": limit, "offset": offset})


@st.cache_data(ttl=5)
def list_alerts(status: Optional[str] = None, severity: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
    """GET /api/alerts"""
    params = {"limit": limit}
    if status:
        params["status"] = status
    if severity:
        params["severity"] = severity
    return _request("GET", "alerts", params=params)


@st.cache_data(ttl=5)
def get_machine_alerts(machine_id: str) -> Dict[str, Any]:
    """GET /api/alerts/<machine_id>"""
    return _request("GET", f"alerts/{machine_id}")


def update_alert_status(alert_id: str, status: str) -> Dict[str, Any]:
    """PATCH /api/alerts/<alert_id>"""
    res = _request("PATCH", f"alerts/{alert_id}", json_data={"status": status})
    st.cache_data.clear()
    return res
