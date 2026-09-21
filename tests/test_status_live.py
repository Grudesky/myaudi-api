import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import server as api_module


@pytest.fixture
def status_client(monkeypatch):
    monkeypatch.setattr(api_module, "AUDI_API_KEY", "test-key")
    monkeypatch.setattr(api_module.client, "ensure_auth", AsyncMock(return_value=True))
    monkeypatch.setattr(api_module.client, "authenticated", True)
    monkeypatch.setattr(api_module.client, "vehicles", [])
    return TestClient(api_module.app)


def test_status_returns_live_after_real_update(status_client, monkeypatch):
    now = time.time()
    monkeypatch.setattr(api_module.client, "_last_live_poll", now)
    live_update = AsyncMock(return_value=(True, 0.0))
    monkeypatch.setattr(
        api_module.client,
        "live_update_vehicles",
        live_update,
    )

    response = status_client.get(
        "/status",
        headers={"X-API-Key": "test-key"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "live"
    assert body["count"] == 0
    assert body["vehicles"] == []
    assert "live_poll_at" in body
    live_update.assert_awaited_once_with(force=False)


def test_status_returns_429_during_live_poll_cooldown(status_client, monkeypatch):
    now = time.time()
    monkeypatch.setattr(api_module.client, "_last_live_poll", now)
    live_update = AsyncMock(return_value=(False, 123.4))
    monkeypatch.setattr(
        api_module.client,
        "live_update_vehicles",
        live_update,
    )

    response = status_client.get(
        "/status",
        headers={"X-API-Key": "test-key"},
    )

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "124"

    body = response.json()
    assert body["status"] == "refresh_blocked"
    assert body["reason"] == "live_poll_cooldown"
    assert body["retry_after_seconds"] == 124
    assert "last_live_poll" in body
    assert "next_live_poll" in body
    assert "vehicles" not in body
    live_update.assert_awaited_once_with(force=False)


def test_status_force_bypasses_live_poll_cooldown(status_client, monkeypatch):
    now = time.time()
    monkeypatch.setattr(api_module.client, "_last_live_poll", now)
    live_update = AsyncMock(return_value=(True, 0.0))
    monkeypatch.setattr(api_module.client, "live_update_vehicles", live_update)

    response = status_client.get(
        "/status?force=true",
        headers={"X-API-Key": "test-key"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "live"
    live_update.assert_awaited_once_with(force=True)


@pytest.mark.parametrize("path", ["/status", "/status?force=true"])
def test_status_live_request_fetches_selective_status_only(
    status_client,
    monkeypatch,
    path,
):
    vehicle = MagicMock()
    vehicle.vin = "WAUTEST"
    vehicle.model = "Q5"
    vehicle.title = "Test vehicle"
    vehicle._fetch_vehicle_data = AsyncMock(return_value=None)
    vehicle._fetch_position = AsyncMock(return_value=None)
    vehicle._fetch_trip = AsyncMock(return_value=None)
    vehicle.get_dashboard.return_value = {}
    monkeypatch.setattr(api_module.client, "vehicles", [vehicle])
    monkeypatch.setattr(api_module.client, "_last_live_poll", 0.0)
    monkeypatch.setattr(api_module.client, "_save_live_poll_state", MagicMock())

    response = status_client.get(
        path,
        headers={"X-API-Key": "test-key"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "live"
    vehicle._fetch_vehicle_data.assert_awaited_once_with(raise_on_error=True)
    vehicle._fetch_position.assert_not_awaited()
    vehicle._fetch_trip.assert_not_awaited()


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500, 502, 503, 504])
@pytest.mark.parametrize("force", [False, True])
def test_status_preserves_upstream_http_error(status_client, monkeypatch, status, force):
    error = api_module.ClientResponseError(
        request_info=MagicMock(), history=(), status=status,
        message="Upstream failure", headers={"Retry-After": "120"},
    )
    vehicle = MagicMock()
    vehicle._fetch_vehicle_data = AsyncMock(side_effect=error)
    monkeypatch.setattr(api_module.client, "vehicles", [vehicle])
    monkeypatch.setattr(api_module.client, "_last_live_poll", 0.0)
    response = status_client.get(
        "/status?force=true" if force else "/status",
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code == status
    assert response.json() == {"detail": f"Audi upstream HTTP {status}"}
    assert response.headers["Retry-After"] == "120"
    assert api_module.client._last_live_poll == 0.0
    assert vehicle._fetch_vehicle_data.await_count == (2 if status == 401 else 1)
