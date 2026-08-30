import time
from unittest.mock import AsyncMock, MagicMock

import pytest

import server as api_module


@pytest.mark.asyncio
async def test_live_update_fetches_only_selective_status_per_vehicle_and_sets_timestamp(monkeypatch):
    client = api_module.AudiClient()
    vehicles = [MagicMock(), MagicMock()]
    for vehicle in vehicles:
        vehicle._fetch_vehicle_data = AsyncMock(return_value=None)
        vehicle._fetch_position = AsyncMock(return_value=None)
        vehicle._fetch_trip = AsyncMock(return_value=None)
    client.vehicles = vehicles
    monkeypatch.setattr(client, "_save_live_poll_state", MagicMock())

    updated, retry_after = await client.live_update_vehicles()

    assert updated is True
    assert retry_after == 0.0
    for vehicle in vehicles:
        vehicle._fetch_vehicle_data.assert_awaited_once_with(raise_on_error=True)
        vehicle._fetch_position.assert_not_awaited()
        vehicle._fetch_trip.assert_not_awaited()
    assert client._last_live_poll > 0


@pytest.mark.asyncio
async def test_live_update_blocks_during_cooldown(monkeypatch):
    client = api_module.AudiClient()
    vehicle = MagicMock()
    vehicle._fetch_vehicle_data = AsyncMock(return_value=None)
    vehicle._fetch_position = AsyncMock(return_value=None)
    vehicle._fetch_trip = AsyncMock(return_value=None)
    client.vehicles = [vehicle]
    client._last_live_poll = time.time()

    updated, retry_after = await client.live_update_vehicles()

    assert updated is False
    assert 0 < retry_after <= api_module.LIVE_POLL_MIN_INTERVAL
    vehicle._fetch_vehicle_data.assert_not_awaited()
    vehicle._fetch_position.assert_not_awaited()
    vehicle._fetch_trip.assert_not_awaited()


@pytest.mark.asyncio
async def test_capability_fetch_uses_existing_live_poll_and_preserves_cooldown(monkeypatch):
    client = api_module.AudiClient()
    vehicle = MagicMock()
    vehicle._fetch_vehicle_data = AsyncMock(return_value=None)
    vehicle._fetch_position = AsyncMock(return_value=None)
    vehicle._fetch_trip = AsyncMock(return_value=None)
    client.vehicles = [vehicle]
    monkeypatch.setattr(client, "_save_live_poll_state", MagicMock())

    updated, retry_after = await client.live_update_vehicles()
    completed_at = client._last_live_poll
    blocked, blocked_retry_after = await client.live_update_vehicles()

    assert updated is True
    assert retry_after == 0.0
    assert blocked is False
    assert 0 < blocked_retry_after <= api_module.LIVE_POLL_MIN_INTERVAL
    assert client._last_live_poll == completed_at
    vehicle._fetch_vehicle_data.assert_awaited_once_with(raise_on_error=True)
    vehicle._fetch_position.assert_not_awaited()


@pytest.mark.asyncio
async def test_forced_live_update_bypasses_cooldown(monkeypatch):
    client = api_module.AudiClient()
    vehicle = MagicMock()
    vehicle._fetch_vehicle_data = AsyncMock(return_value=None)
    vehicle._fetch_position = AsyncMock(return_value=None)
    vehicle._fetch_trip = AsyncMock(return_value=None)
    client.vehicles = [vehicle]
    client._last_live_poll = time.time()
    monkeypatch.setattr(client, "_save_live_poll_state", MagicMock())

    updated, retry_after = await client.live_update_vehicles(force=True)

    assert updated is True
    assert retry_after == 0.0
    vehicle._fetch_vehicle_data.assert_awaited_once_with(raise_on_error=True)
    vehicle._fetch_position.assert_not_awaited()
    vehicle._fetch_trip.assert_not_awaited()


@pytest.mark.asyncio
async def test_status_only_live_update_preserves_last_known_position(monkeypatch):
    client = api_module.AudiClient()
    vehicle = MagicMock()
    last_known_position = {
        "latitude": 47.6205,
        "longitude": -122.3493,
        "timestamp": "2026-08-30T12:00:00Z",
    }
    vehicle._position = last_known_position
    vehicle._fetch_vehicle_data = AsyncMock(return_value=None)
    vehicle._fetch_position = AsyncMock(return_value=None)
    client.vehicles = [vehicle]
    monkeypatch.setattr(client, "_save_live_poll_state", MagicMock())

    updated, retry_after = await client.live_update_vehicles()

    assert updated is True
    assert retry_after == 0.0
    assert vehicle._position is last_known_position
    vehicle._fetch_position.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_live_update_does_not_advance_timestamp(monkeypatch):
    client = api_module.AudiClient()
    vehicle = MagicMock()
    vehicle._fetch_vehicle_data = AsyncMock(side_effect=RuntimeError("Audi failed"))
    client.vehicles = [vehicle]
    client._last_live_poll = 0.0

    with pytest.raises(RuntimeError, match="Audi failed"):
        await client.live_update_vehicles()

    assert client._last_live_poll == 0.0

def test_live_poll_state_round_trip(tmp_path, monkeypatch):
    state_file = tmp_path / "live-poll.json"
    monkeypatch.setattr(api_module, "LIVE_POLL_STATE_FILE", state_file)

    client = api_module.AudiClient()
    client._last_live_poll = 1234567890.5
    client._save_live_poll_state()

    restored = api_module.AudiClient()
    restored._load_live_poll_state()

    assert restored._last_live_poll == 1234567890.5

@pytest.mark.asyncio
async def test_live_update_reauthenticates_once_after_401(monkeypatch):
    client = api_module.AudiClient()

    error_401 = api_module.ClientResponseError(
        request_info=MagicMock(),
        history=(),
        status=401,
        message="Unauthorized",
    )

    vehicle = MagicMock()
    vehicle._fetch_vehicle_data = AsyncMock(
        side_effect=[error_401, None]
    )
    vehicle._fetch_position = AsyncMock(return_value=None)
    vehicle._fetch_trip = AsyncMock(return_value=None)
    client.vehicles = [vehicle]

    ensure_auth = AsyncMock(return_value=True)
    monkeypatch.setattr(client, "ensure_auth", ensure_auth)
    monkeypatch.setattr(client, "_save_live_poll_state", MagicMock())

    updated, retry_after = await client.live_update_vehicles()

    assert updated is True
    assert retry_after == 0.0
    assert vehicle._fetch_vehicle_data.await_count == 2
    vehicle._fetch_position.assert_not_awaited()
    vehicle._fetch_trip.assert_not_awaited()
    ensure_auth.assert_awaited_once()
    assert client._last_live_poll > 0
