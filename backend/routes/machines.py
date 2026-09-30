"""
machines.py - REST route handlers for Machine management endpoints.
"""

from flask import Blueprint, request
from backend.utils.responses import success_response
from backend.utils.errors import APIError
from backend.services.machine_service import (
    create_machine,
    get_machine,
    list_machines
)

machines_bp = Blueprint("machines", __name__)


@machines_bp.route("/machines", methods=["GET"])
def get_machines_list():
    """
    GET /api/machines
    Returns list of registered machines.
    """
    machines = list_machines()
    return success_response(data={"machines": machines}, status_code=200)


@machines_bp.route("/machines/<machine_id>", methods=["GET"])
def get_single_machine(machine_id: str):
    """
    GET /api/machines/<machine_id>
    Returns details for a specific machine.
    """
    machine = get_machine(machine_id)
    return success_response(data=machine, status_code=200)


@machines_bp.route("/machines", methods=["POST"])
def post_create_machine():
    """
    POST /api/machines
    Creates a new machine.
    Body: { "machine_id": "...", "name": "...", "description": "..." }
    """
    if not request.is_json:
        raise APIError("VALIDATION_ERROR", "Request body must be JSON.", status_code=400)

    body = request.get_json() or {}
    machine_id = body.get("machine_id")
    name = body.get("name")
    description = body.get("description")

    created = create_machine(machine_id=machine_id, name=name, description=description)
    return success_response(data=created, status_code=201)
