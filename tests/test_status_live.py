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
