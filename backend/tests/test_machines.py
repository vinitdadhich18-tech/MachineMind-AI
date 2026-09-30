"""
test_machines.py - Unit and integration tests for Machines CRUD endpoints.
"""


def test_create_and_get_machine(client):
    """Verifies machine creation, retrieval, and listing."""
    # 1. Create machine
    res_create = client.post(
        "/api/machines",
        json={"machine_id": "test_rig_01", "name": "Test Rig 01", "description": "NASA IMS set 2 rig"}
    )
    assert res_create.status_code == 201
    payload = res_create.get_json()
    assert payload["success"] is True
    assert payload["data"]["machine_id"] == "test_rig_01"
    assert payload["data"]["status"] == "no_data"

    # 2. Get single machine
    res_get = client.get("/api/machines/test_rig_01")
    assert res_get.status_code == 200
    assert res_get.get_json()["data"]["name"] == "Test Rig 01"

    # 3. List machines
    res_list = client.get("/api/machines")
    assert res_list.status_code == 200
    machines = res_list.get_json()["data"]["machines"]
    assert len(machines) == 1
    assert machines[0]["machine_id"] == "test_rig_01"


def test_machine_creation_validation_and_duplicate(client):
    """Verifies validation failure codes (400, 409, 404)."""
    # Create initial machine
    client.post("/api/machines", json={"machine_id": "rig_alpha", "name": "Rig Alpha"})

    # Duplicate -> 409
    res_dup = client.post("/api/machines", json={"machine_id": "rig_alpha", "name": "Rig Alpha Duplicate"})
    assert res_dup.status_code == 409
    assert res_dup.get_json()["error"]["code"] == "MACHINE_EXISTS"

    # Invalid machine_id format -> 400
    res_invalid = client.post("/api/machines", json={"machine_id": "invalid id spaces!", "name": "Bad"})
    assert res_invalid.status_code == 400
    assert res_invalid.get_json()["error"]["code"] == "VALIDATION_ERROR"

    # Get unknown machine -> 404
    res_404 = client.get("/api/machines/unknown_machine_99")
    assert res_404.status_code == 404
    assert res_404.get_json()["error"]["code"] == "MACHINE_NOT_FOUND"
