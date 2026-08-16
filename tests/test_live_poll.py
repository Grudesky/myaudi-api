import time
from unittest.mock import AsyncMock, MagicMock

import pytest

import server as api_module


@pytest.mark.asyncio
async def test_live_update_performs_fetch_and_sets_timestamp(monkeypatch):
    client = api_module.AudiClient()
    vehicle = MagicMock()
    vehicle._fetch_vehicle_data = AsyncMock(return_value=None)
    client.vehicles = [vehicle]

    updated, retry_after = await client.live_update_vehicles()

    assert updated is True
    assert retry_after == 0.0
    vehicle._fetch_vehicle_data.assert_awaited_once_with(raise_on_error=True)
    assert client._last_live_poll > 0


@pytest.mark.asyncio
async def test_live_update_blocks_during_cooldown(monkeypatch):
    client = api_module.AudiClient()
    vehicle = MagicMock()
    vehicle._fetch_vehicle_data = AsyncMock(return_value=None)
    client.vehicles = [vehicle]
    client._last_live_poll = time.time()

    updated, retry_after = await client.live_update_vehicles()

    assert updated is False
    assert 0 < retry_after <= api_module.LIVE_POLL_MIN_INTERVAL
    vehicle._fetch_vehicle_data.assert_not_awaited()


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
