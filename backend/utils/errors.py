"""
errors.py - Custom exception classes and global error handlers for MachineMind AI Flask Backend.
"""

import logging
from flask import Flask
from backend.utils.responses import error_response

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Base exception class for explicit API errors."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        details: dict = None
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


def register_error_handlers(app: Flask) -> None:
    """Registers error handlers to prevent raw HTML responses and internal tracebacks from leaking."""

    @app.errorhandler(APIError)
    def handle_api_error(err: APIError):
        return error_response(
            code=err.code,
            message=err.message,
            details=err.details,
            status_code=err.status_code
        )

    @app.errorhandler(400)
    def bad_request(err):
        return error_response("VALIDATION_ERROR", "Bad request.", status_code=400)

    @app.errorhandler(404)
    def not_found(err):
        return error_response("NOT_FOUND", "Requested resource not found.", status_code=404)

    @app.errorhandler(405)
    def method_not_allowed(err):
        return error_response("METHOD_NOT_ALLOWED", "HTTP method not allowed.", status_code=405)

    @app.errorhandler(413)
    def payload_too_large(err):
        return error_response("PAYLOAD_TOO_LARGE", "Uploaded file exceeds size limit.", status_code=413)

    @app.errorhandler(415)
    def unsupported_media_type(err):
        return error_response("UNSUPPORTED_FILE_TYPE", "Unsupported media or file type.", status_code=415)

    @app.errorhandler(422)
    def unprocessable_entity(err):
        return error_response("VALIDATION_ERROR", "Unprocessable request parameters.", status_code=422)

    @app.errorhandler(500)
    def internal_server_error(err):
        logger.error(f"Internal Server Error: {err}", exc_info=True)
        return error_response("INTERNAL_ERROR", "An unexpected internal server error occurred.", status_code=500)
