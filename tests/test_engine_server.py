from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import server as api_module
from audi_connect.exceptions import ActionPersistenceError


@pytest.fixture
def engine_client(monkeypatch):
    api_module.limiter.reset()
    vehicle = MagicMock()
    vehicle.vin = "WAUTEST"
    vehicle.start_engine = AsyncMock(return_value="request-start")
    vehicle.stop_engine = AsyncMock(return_value="request-stop")
    vehicle.get_engine_action_status = AsyncMock(
        return_value={
            "status": "in_progress",
            "request_id": "request-start",
            "upstream_status": "in_progress",
        }
    )
    vehicle.active_engine_action = {
        "local_action_id": "11111111-1111-1111-1111-111111111111",
        "vin": "WAUTEST",
        "action": "engine_start",
        "status": "ambiguous",
        "request_id": None,
        "created_at": "2026-08-29T12:00:00+00:00",
        "updated_at": "2026-08-29T12:00:01+00:00",
    }
    vehicle.recover_engine_action = AsyncMock(
        return_value={
            **vehicle.active_engine_action,
            "status": "operator_recovered",
        }
    )
    monkeypatch.setattr(api_module, "AUDI_API_KEY", "test-key")
    monkeypatch.setattr(
        api_module,
        "AUDI_ENGINE_CONTROL_VINS",
        frozenset({"WAUTEST"}),
    )
    monkeypatch.setattr(
        api_module.client,
        "ensure_auth",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(api_module.client, "get_vehicle", MagicMock(return_value=vehicle))
    monkeypatch.setattr(api_module.client, "vehicles", [vehicle])
    monkeypatch.setattr(api_module.client, "authenticated", True)
    monkeypatch.setattr(api_module.client, "invalidate_cache", MagicMock())
    monkeypatch.setattr(api_module.client, "update_vehicles", AsyncMock())
    monkeypatch.setattr(api_module.client, "live_update_vehicles", AsyncMock())
    monkeypatch.setattr(api_module.client, "_last_live_poll", 1234567890.0)
    monkeypatch.setattr(api_module.client, "_save_live_poll_state", MagicMock())
    return TestClient(api_module.app), vehicle


def assert_polling_untouched(vehicle):
    assert api_module.client._last_live_poll == 1234567890.0
    api_module.client.invalidate_cache.assert_not_called()
    api_module.client.update_vehicles.assert_not_awaited()
    api_module.client.live_update_vehicles.assert_not_awaited()
    api_module.client._save_live_poll_state.assert_not_called()
    vehicle._fetch_vehicle_data.assert_not_called()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, frozenset()),
        ("", frozenset()),
        ("   ", frozenset()),
        (
            "  wautest , WAUOTHER  ,wauthird ",
            frozenset({"WAUTEST", "WAUOTHER", "WAUTHIRD"}),
        ),
    ],
)
def test_engine_control_vin_allowlist_parsing(value, expected):
    assert api_module.parse_engine_control_vins(value) == expected


@pytest.mark.asyncio
async def test_login_applies_engine_allowlist_without_status_poll(monkeypatch):
    auth = MagicMock()
    auth.login = AsyncMock(
        return_value=[{"vin": "wautest"}, {"vin": "WAUOTHER"}]
    )
    monkeypatch.setattr(api_module, "AUDI_ENGINE_CONTROL_VINS", frozenset({"WAUTEST"}))
    monkeypatch.setattr(api_module, "AudiAPI", MagicMock())
    monkeypatch.setattr(api_module, "AudiAuth", MagicMock(return_value=auth))
    client = api_module.AudiClient()
    client._session = MagicMock()

    assert await client.login() is True

    assert client.vehicles[0]._engine_control_enabled is True
    assert client.vehicles[1]._engine_control_enabled is False
    auth.get_stored_vehicle_data.assert_not_called()


@pytest.mark.parametrize(
    ("path", "method_name"),
    [
        ("/WAUTEST/engine/start", "start_engine"),
        ("/WAUTEST/engine/stop", "stop_engine"),
    ],
)
def test_configured_vin_works_without_capability_snapshot_after_restart(
    engine_client,
    path,
    method_name,
):
    test_client, vehicle = engine_client
    vehicle.capability_ids = ()

    response = test_client.post(
        path,
        headers={"X-API-Key": "test-key"},
    )

    assert response.status_code == 200
    getattr(vehicle, method_name).assert_awaited_once_with()
    assert_polling_untouched(vehicle)


def test_configured_vin_works_with_engine_control_capability(engine_client):
    test_client, vehicle = engine_client
    vehicle.capability_ids = ("engineControl",)

    response = test_client.post(
        "/WAUTEST/engine/start",
        headers={"X-API-Key": "test-key"},
    )

    assert response.status_code == 200
    vehicle.start_engine.assert_awaited_once_with()
    assert_polling_untouched(vehicle)


def test_engine_control_vin_comparison_is_case_insensitive(engine_client):
    test_client, vehicle = engine_client

    response = test_client.post(
        "/wautest/engine/start",
        headers={"X-API-Key": "test-key"},
    )

    assert response.status_code == 200
    vehicle.start_engine.assert_awaited_once_with()


@pytest.mark.parametrize("allowlist", [frozenset(), frozenset({"WAUOTHER"})])
@pytest.mark.parametrize("action", ["start", "stop"])
def test_unconfigured_or_empty_allowlist_rejects_locally(
    engine_client,
    monkeypatch,
    allowlist,
    action,
):
    test_client, vehicle = engine_client
    monkeypatch.setattr(api_module, "AUDI_ENGINE_CONTROL_VINS", allowlist)

    response = test_client.post(
        f"/WAUTEST/engine/{action}",
        headers={"X-API-Key": "test-key"},
    )

    assert response.status_code == 403
    assert response.json() == {
        "detail": "Engine control is not enabled for this VIN"
    }
    api_module.client.ensure_auth.assert_not_awaited()
    api_module.client.get_vehicle.assert_not_called()
    vehicle.start_engine.assert_not_awaited()
    vehicle.stop_engine.assert_not_awaited()
    assert_polling_untouched(vehicle)


def test_engine_start_endpoint_returns_request_id_without_refresh(engine_client):
    test_client, vehicle = engine_client

    response = test_client.post(
        "/WAUTEST/engine/start",
        headers={"X-API-Key": "test-key"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "sent",
        "action": "engine_start",
        "vin": "WAUTEST",
        "request_id": "request-start",
    }
    vehicle.start_engine.assert_awaited_once_with()
    assert_polling_untouched(vehicle)


def test_engine_stop_endpoint_returns_request_id_without_refresh(engine_client):
    test_client, vehicle = engine_client

    response = test_client.post(
        "/WAUTEST/engine/stop",
        headers={"X-API-Key": "test-key"},
    )

    assert response.status_code == 200
    assert response.json()["request_id"] == "request-stop"
    assert response.json()["status"] == "sent"
    vehicle.stop_engine.assert_awaited_once_with()
    assert_polling_untouched(vehicle)


def test_engine_start_fails_closed_when_durable_state_is_unavailable(engine_client):
    test_client, vehicle = engine_client
    vehicle.start_engine.side_effect = ActionPersistenceError(
        "Durable engine-action storage is required"
    )

    response = test_client.post(
        "/WAUTEST/engine/start",
        headers={"X-API-Key": "test-key"},
    )

    assert response.status_code == 503
    vehicle.start_engine.assert_awaited_once_with()
    assert_polling_untouched(vehicle)


def test_action_status_endpoint_returns_structured_state_without_refresh(engine_client):
    test_client, vehicle = engine_client

    response = test_client.get(
        "/WAUTEST/actions/request-start",
        headers={"X-API-Key": "test-key"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "in_progress",
        "request_id": "request-start",
        "upstream_status": "in_progress",
        "vin": "WAUTEST",
    }
    vehicle.get_engine_action_status.assert_awaited_once_with("request-start")
    assert_polling_untouched(vehicle)


def test_active_action_endpoint_returns_secret_free_guard(engine_client):
    test_client, vehicle = engine_client

    response = test_client.get(
        "/WAUTEST/engine/action",
        headers={"X-API-Key": "test-key"},
    )

    assert response.status_code == 200
    assert response.json()["active_action"]["status"] == "ambiguous"
    assert "spin" not in response.text
    assert "Authorization" not in response.text
    assert_polling_untouched(vehicle)


def test_explicit_recovery_requires_confirmation_and_never_polls(engine_client):
    test_client, vehicle = engine_client
    local_id = "11111111-1111-1111-1111-111111111111"

    rejected = test_client.post(
        f"/WAUTEST/engine/actions/{local_id}/recover",
        headers={"X-API-Key": "test-key"},
    )
    assert rejected.status_code == 400
    vehicle.recover_engine_action.assert_not_awaited()

    response = test_client.post(
        f"/WAUTEST/engine/actions/{local_id}/recover?confirm=true",
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "status": "operator_recovered",
        "vin": "WAUTEST",
        "action": "engine_start",
        "local_action_id": local_id,
        "request_id": None,
    }
    vehicle.recover_engine_action.assert_awaited_once_with(local_id)
    vehicle.start_engine.assert_not_awaited()
    vehicle.stop_engine.assert_not_awaited()
    assert_polling_untouched(vehicle)


@pytest.mark.parametrize("path", ["/WAUTEST/engine/start", "/WAUTEST/engine/stop"])
def test_engine_endpoints_require_api_key(engine_client, path):
    test_client, _ = engine_client

    response = test_client.post(path)

    assert response.status_code == 401
