"""Deferred parking-position refresh policy tests (all upstream calls mocked)."""

import asyncio
import json

from unittest.mock import AsyncMock, MagicMock

import pytest

import server as api_module
from audi_connect.client import ParkingPositionResult
from audi_connect.position_state import PositionStateStore, normalize_position
from audi_connect.vehicle import LegacyPositionUpdate


OLD_POSITION = {
    "lat": 47.0,
    "lon": -122.0,
    "carCapturedTimestamp": "2026-08-01T12:00:00Z",
}
NEW_POSITION = {
    "data": {
        "lat": 48.0,
        "lon": -123.0,
        "carCapturedTimestamp": "2026-08-02T12:00:00Z",
    }
}


class FakeVehicle:
    def __init__(self, odometer=100, position=OLD_POSITION):
        self.vin = "WAUTEST1234567890"
        self._mileage = odometer
        self._mileage_timestamp = "2026-08-01T13:00:00Z"
        self._position = position
        self._position_failed = False
        self._position_fetched = position is not None
        self._fetch_vehicle_data = AsyncMock(return_value=None)
        self._fetch_position_candidate = AsyncMock(
            return_value=ParkingPositionResult(
                outcome="success",
                position=NEW_POSITION,
                http_status=200,
            )
        )
        self._fetch_trip = AsyncMock(return_value=None)

    @property
    def mileage(self):
        return self._mileage

    @property
    def mileage_timestamp(self):
        return self._mileage_timestamp

    def _restore_position(self, position):
        self._position = position
        self._position_failed = False
        self._position_fetched = True

    def _accept_position(self, position):
        self._restore_position(position)


@pytest.fixture
def position_client(tmp_path, monkeypatch):
    monkeypatch.setattr(
        api_module,
        "POSITION_STATE_FILE",
        tmp_path / "position-state.json",
    )
    monkeypatch.setattr(api_module, "LIVE_POLL_MIN_INTERVAL", 0)
    client = api_module.AudiClient()
    monkeypatch.setattr(client, "_save_live_poll_state", MagicMock())
    return client


async def poll(client, vehicle, *, force=False):
    client.vehicles = [vehicle]
    return await client.live_update_vehicles(force=force)


@pytest.mark.asyncio
async def test_three_idle_days_make_no_position_or_trip_requests(position_client):
    vehicle = FakeVehicle()

    for _ in range(3 * 24 * 4):
        updated, _ = await poll(position_client, vehicle)
        assert updated is True

    assert vehicle._fetch_vehicle_data.await_count == 288
    vehicle._fetch_position_candidate.assert_not_awaited()
    vehicle._fetch_trip.assert_not_awaited()


@pytest.mark.asyncio
async def test_increase_defers_then_stable_poll_accepts_newer_position(position_client):
    vehicle = FakeVehicle()
    await poll(position_client, vehicle)

    vehicle._mileage = 101
    await poll(position_client, vehicle)
    state = position_client._position_state_store.state_for(vehicle.vin)
    assert state["pending_odometer"] == 101
    assert state["attempt_count"] == 0
    vehicle._fetch_position_candidate.assert_not_awaited()
    assert vehicle._position == OLD_POSITION

    await poll(position_client, vehicle)
    state = position_client._position_state_store.state_for(vehicle.vin)
    vehicle._fetch_position_candidate.assert_awaited_once()
    assert vehicle._position == {
        "lat": 48.0,
        "lon": -123.0,
        "carCapturedTimestamp": "2026-08-02T12:00:00Z",
    }
    assert state["position_odometer"] == 101
    assert state["pending_odometer"] is None


@pytest.mark.asyncio
async def test_forced_polls_never_detect_or_accelerate_pending_refresh(position_client):
    vehicle = FakeVehicle()
    await poll(position_client, vehicle)

    vehicle._mileage = 101
    await poll(position_client, vehicle, force=True)
    await poll(position_client, vehicle, force=True)
    assert position_client._position_state_store.state_for(vehicle.vin)["pending_odometer"] is None
    vehicle._fetch_position_candidate.assert_not_awaited()

    await poll(position_client, vehicle)
    vehicle._fetch_position_candidate.assert_not_awaited()
    await poll(position_client, vehicle, force=True)
    vehicle._fetch_position_candidate.assert_not_awaited()
    await poll(position_client, vehicle)
    vehicle._fetch_position_candidate.assert_awaited_once()


@pytest.mark.asyncio
async def test_multiple_increases_coalesce_before_one_attempt(position_client):
    vehicle = FakeVehicle()
    await poll(position_client, vehicle)
    for odometer in (101, 102, 103):
        vehicle._mileage = odometer
        await poll(position_client, vehicle)
        vehicle._fetch_position_candidate.assert_not_awaited()

    state = position_client._position_state_store.state_for(vehicle.vin)
    assert state["pending_odometer"] == 103
    await poll(position_client, vehicle)
    vehicle._fetch_position_candidate.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        ParkingPositionResult(outcome="unavailable", http_status=204),
        ParkingPositionResult(outcome="unsupported", http_status=404),
        ParkingPositionResult(outcome="rate_limited", http_status=429),
        ParkingPositionResult(outcome="server_error", http_status=502),
        ParkingPositionResult(outcome="transport_error", error_type="RequestTimeoutError"),
    ],
)
async def test_unsuccessful_candidate_gets_one_retry_then_stops(
    position_client,
    result,
):
    vehicle = FakeVehicle()
    vehicle._fetch_position_candidate.return_value = result
    await poll(position_client, vehicle)
    vehicle._mileage = 101
    await poll(position_client, vehicle)
    await poll(position_client, vehicle)

    assert vehicle._position is OLD_POSITION
    assert position_client._position_state_store.state_for(vehicle.vin)["attempt_count"] == 1
    await poll(position_client, vehicle)
    for _ in range(8):
        await poll(position_client, vehicle)
    assert vehicle._fetch_position_candidate.await_count == 2
    assert position_client._position_state_store.state_for(vehicle.vin)["attempt_count"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "timestamp",
    ["2026-08-01T12:00:00Z", "2026-07-31T12:00:00Z"],
)
async def test_stale_or_equal_candidate_is_rejected_after_two_attempts(
    position_client,
    timestamp,
):
    vehicle = FakeVehicle()
    vehicle._fetch_position_candidate.return_value = ParkingPositionResult(
        outcome="success",
        position={"lat": 49.0, "lon": 99.0, "carCapturedTimestamp": timestamp},
        http_status=200,
    )
    await poll(position_client, vehicle)
    vehicle._mileage = 101
    await poll(position_client, vehicle)
    await poll(position_client, vehicle)
    await poll(position_client, vehicle)
    await poll(position_client, vehicle)

    assert vehicle._position is OLD_POSITION
    assert vehicle._fetch_position_candidate.await_count == 2
    state = position_client._position_state_store.state_for(vehicle.vin)
    assert state["pending_odometer"] == 101
    assert state["attempt_count"] == 2


@pytest.mark.asyncio
async def test_candidate_exception_does_not_fail_status_or_exceed_two(position_client):
    vehicle = FakeVehicle()
    vehicle._fetch_position_candidate.side_effect = TimeoutError("mock timeout")
    await poll(position_client, vehicle)
    vehicle._mileage = 101
    await poll(position_client, vehicle)
    updated, _ = await poll(position_client, vehicle)
    assert updated is True
    await poll(position_client, vehicle)
    await poll(position_client, vehicle)
    assert vehicle._fetch_position_candidate.await_count == 2
    assert vehicle._position is OLD_POSITION


@pytest.mark.asyncio
async def test_failed_first_attempt_can_succeed_on_second(position_client):
    vehicle = FakeVehicle()
    vehicle._fetch_position_candidate.side_effect = [
        ParkingPositionResult(outcome="unavailable", http_status=204),
        ParkingPositionResult(outcome="success", position=NEW_POSITION, http_status=200),
    ]
    await poll(position_client, vehicle)
    vehicle._mileage = 101
    await poll(position_client, vehicle)
    await poll(position_client, vehicle)
    await poll(position_client, vehicle)

    state = position_client._position_state_store.state_for(vehicle.vin)
    assert vehicle._fetch_position_candidate.await_count == 2
    assert state["pending_odometer"] is None
    assert state["attempt_count"] == 0
    assert vehicle._position["lat"] == 48.0


@pytest.mark.asyncio
async def test_increase_after_exhaustion_starts_new_episode(position_client):
    vehicle = FakeVehicle()
    vehicle._fetch_position_candidate.return_value = ParkingPositionResult(
        outcome="unavailable",
        http_status=204,
    )
    await poll(position_client, vehicle)
    vehicle._mileage = 101
    await poll(position_client, vehicle)
    await poll(position_client, vehicle)
    await poll(position_client, vehicle)
    assert vehicle._fetch_position_candidate.await_count == 2

    vehicle._mileage = 102
    await poll(position_client, vehicle)
    state = position_client._position_state_store.state_for(vehicle.vin)
    assert state["pending_odometer"] == 102
    assert state["attempt_count"] == 0
    assert vehicle._fetch_position_candidate.await_count == 2

    await poll(position_client, vehicle)
    assert vehicle._fetch_position_candidate.await_count == 3
    assert position_client._position_state_store.state_for(vehicle.vin)[
        "attempt_count"
    ] == 1


@pytest.mark.asyncio
async def test_continued_movement_postpones_second_attempt_until_stable(position_client):
    vehicle = FakeVehicle()
    vehicle._fetch_position_candidate.return_value = ParkingPositionResult(
        outcome="unavailable",
        http_status=204,
    )
    await poll(position_client, vehicle)
    vehicle._mileage = 101
    await poll(position_client, vehicle)
    await poll(position_client, vehicle)
    assert vehicle._fetch_position_candidate.await_count == 1

    for odometer in (102, 103):
        vehicle._mileage = odometer
        await poll(position_client, vehicle)
        assert vehicle._fetch_position_candidate.await_count == 1

    await poll(position_client, vehicle)
    assert vehicle._fetch_position_candidate.await_count == 2


def _seed_pending_store(path):
    store = PositionStateStore(path)
    store.observe(
        "WAUTEST1234567890",
        100,
        "2026-08-01T12:00:00Z",
        current_position=OLD_POSITION,
    )
    store.observe("WAUTEST1234567890", 101, "2026-08-01T12:15:00Z")
    return store


def test_restart_after_pending_before_claim_preserves_both_attempts(tmp_path):
    path = tmp_path / "position-state.json"
    _seed_pending_store(path)

    restarted = PositionStateStore(path)
    assert restarted.state_for("WAUTEST1234567890")["attempt_count"] == 0
    assert restarted.observe(
        "WAUTEST1234567890", 101, "2026-08-01T12:30:00Z"
    ) is True
    assert restarted.state_for("WAUTEST1234567890")["attempt_count"] == 1


def test_restart_after_claim_before_get_consumes_first_attempt(tmp_path):
    path = tmp_path / "position-state.json"
    store = _seed_pending_store(path)
    assert store.observe("WAUTEST1234567890", 101, "2026-08-01T12:30:00Z") is True

    restarted = PositionStateStore(path)
    assert restarted.state_for("WAUTEST1234567890")["attempt_count"] == 1
    assert restarted.observe(
        "WAUTEST1234567890", 101, "2026-08-01T12:45:00Z"
    ) is True
    assert restarted.observe(
        "WAUTEST1234567890", 101, "2026-08-01T13:00:00Z"
    ) is False


def test_restart_after_get_before_acceptance_consumes_attempt(tmp_path):
    path = tmp_path / "position-state.json"
    store = _seed_pending_store(path)
    assert store.observe("WAUTEST1234567890", 101, "2026-08-01T12:30:00Z") is True
    # Simulate a completed GET followed by process loss before store.accept().

    restarted = PositionStateStore(path)
    assert restarted.state_for("WAUTEST1234567890")["attempt_count"] == 1
    assert restarted.observe(
        "WAUTEST1234567890", 101, "2026-08-01T12:45:00Z"
    ) is True


def test_restart_after_acceptance_restores_new_position_without_attempt(tmp_path):
    path = tmp_path / "position-state.json"
    store = _seed_pending_store(path)
    assert store.observe("WAUTEST1234567890", 101, "2026-08-01T12:30:00Z") is True
    store.accept("WAUTEST1234567890", NEW_POSITION, 101)

    restarted = PositionStateStore(path)
    state = restarted.state_for("WAUTEST1234567890")
    assert state["accepted_position"]["lat"] == 48.0
    assert state["pending_odometer"] is None
    assert restarted.observe(
        "WAUTEST1234567890", 101, "2026-08-01T12:45:00Z"
    ) is False


def test_retry_after_blocks_second_attempt_until_expiry(tmp_path):
    path = tmp_path / "position-state.json"
    store = _seed_pending_store(path)
    assert store.observe("WAUTEST1234567890", 101, "2026-08-01T12:30:00Z") is True
    store.defer_retry("WAUTEST1234567890", "2026-08-01T13:30:00Z")

    assert store.observe(
        "WAUTEST1234567890", 101, "2026-08-01T12:45:00Z"
    ) is False
    assert store.state_for("WAUTEST1234567890")["attempt_count"] == 1
    assert store.observe(
        "WAUTEST1234567890", 101, "2026-08-01T13:30:00Z"
    ) is True
    assert store.state_for("WAUTEST1234567890")["attempt_count"] == 2


@pytest.mark.asyncio
async def test_force_poll_cannot_change_pending_attempt_state(position_client):
    vehicle = FakeVehicle()
    vehicle._fetch_position_candidate.return_value = ParkingPositionResult(
        outcome="unavailable",
        http_status=204,
    )
    await poll(position_client, vehicle)
    vehicle._mileage = 101
    await poll(position_client, vehicle)
    await poll(position_client, vehicle)
    before = position_client._position_state_store.state_for(vehicle.vin)

    vehicle._mileage = 102
    for _ in range(3):
        await poll(position_client, vehicle, force=True)

    assert position_client._position_state_store.state_for(vehicle.vin) == before
    assert vehicle._fetch_position_candidate.await_count == 1


@pytest.mark.asyncio
async def test_429_retry_after_is_persisted_and_blocks_second_attempt(position_client):
    vehicle = FakeVehicle()
    vehicle._fetch_position_candidate.return_value = ParkingPositionResult(
        outcome="rate_limited",
        http_status=429,
        retry_not_before="2099-01-01T00:00:00+00:00",
    )
    await poll(position_client, vehicle)
    vehicle._mileage = 101
    await poll(position_client, vehicle)
    await poll(position_client, vehicle)

    for _ in range(4):
        await poll(position_client, vehicle)

    state = position_client._position_state_store.state_for(vehicle.vin)
    assert state["attempt_count"] == 1
    assert state["retry_not_before"] == "2099-01-01T00:00:00+00:00"
    vehicle._fetch_position_candidate.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancellation_during_get_consumes_attempt_but_not_episode(tmp_path, monkeypatch):
    state_file = tmp_path / "position-state.json"
    monkeypatch.setattr(api_module, "POSITION_STATE_FILE", state_file)
    monkeypatch.setattr(api_module, "LIVE_POLL_MIN_INTERVAL", 0)
    first = api_module.AudiClient()
    monkeypatch.setattr(first, "_save_live_poll_state", MagicMock())
    vehicle = FakeVehicle()
    await poll(first, vehicle)
    vehicle._mileage = 101
    await poll(first, vehicle)
    vehicle._fetch_position_candidate.side_effect = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await poll(first, vehicle)

    assert first._position_state_store.state_for(vehicle.vin)["attempt_count"] == 1
    restarted = api_module.AudiClient()
    monkeypatch.setattr(restarted, "_save_live_poll_state", MagicMock())
    restored_vehicle = FakeVehicle(odometer=101, position=None)
    restarted.vehicles = [restored_vehicle]
    restarted._restore_positions()
    await poll(restarted, restored_vehicle)
    restored_vehicle._fetch_position_candidate.assert_awaited_once()
    assert restarted._position_state_store.state_for(restored_vehicle.vin)[
        "attempt_count"
    ] == 0


@pytest.mark.asyncio
async def test_legacy_success_reconciles_and_prevents_automatic_duplicate(position_client):
    vehicle = FakeVehicle()
    await poll(position_client, vehicle)
    vehicle._mileage = 101
    await poll(position_client, vehicle)
    vehicle.update = AsyncMock(
        return_value=LegacyPositionUpdate(position=NEW_POSITION, odometer=101)
    )

    await position_client.update_vehicles(force=True)

    state = position_client._position_state_store.state_for(vehicle.vin)
    assert state["accepted_position"]["lat"] == 48.0
    assert state["position_odometer"] == 101
    assert state["pending_odometer"] is None
    await poll(position_client, vehicle)
    vehicle._fetch_position_candidate.assert_not_awaited()

    restarted = api_module.AudiClient()
    restored_vehicle = FakeVehicle(odometer=101, position=None)
    restarted.vehicles = [restored_vehicle]
    restarted._restore_positions()
    assert restored_vehicle._position["lat"] == 48.0


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_result", [None, OLD_POSITION])
async def test_legacy_failed_or_stale_result_preserves_accepted_position(
    position_client,
    legacy_result,
):
    vehicle = FakeVehicle()
    await poll(position_client, vehicle)
    vehicle.update = AsyncMock(
        return_value=LegacyPositionUpdate(position=legacy_result, odometer=100)
    )

    await position_client.update_vehicles(force=True)

    assert vehicle._position == OLD_POSITION
    assert position_client._position_state_store.state_for(vehicle.vin)[
        "accepted_position"
    ] == OLD_POSITION


@pytest.mark.asyncio
async def test_legacy_persistence_failure_never_installs_candidate(
    position_client,
    monkeypatch,
):
    vehicle = FakeVehicle()
    await poll(position_client, vehicle)
    before = position_client._position_state_store.state_for(vehicle.vin)
    vehicle.update = AsyncMock(
        return_value=LegacyPositionUpdate(position=NEW_POSITION, odometer=101)
    )
    monkeypatch.setattr(
        position_client._position_state_store,
        "reconcile",
        MagicMock(side_effect=OSError("disk unavailable")),
    )

    await position_client.update_vehicles(force=True)

    assert vehicle._position == OLD_POSITION
    assert position_client._position_state_store.state_for(vehicle.vin) == before


@pytest.mark.asyncio
async def test_legacy_position_without_contemporaneous_odometer_sets_baseline(
    position_client,
):
    vehicle = FakeVehicle()
    await poll(position_client, vehicle)
    vehicle._mileage = 101
    await poll(position_client, vehicle)
    vehicle.update = AsyncMock(
        return_value=LegacyPositionUpdate(position=NEW_POSITION, odometer=None)
    )

    await position_client.update_vehicles(force=True)

    state = position_client._position_state_store.state_for(vehicle.vin)
    assert state["accepted_position"]["lat"] == 48.0
    assert state["position_odometer"] is None
    assert state["pending_odometer"] is None
    assert vehicle._position["lat"] == 48.0

    await poll(position_client, vehicle)
    await poll(position_client, vehicle)
    vehicle._fetch_position_candidate.assert_not_awaited()
    assert position_client._position_state_store.state_for(vehicle.vin)[
        "position_odometer"
    ] == 101


@pytest.mark.asyncio
async def test_restart_restores_position_and_pending_state(tmp_path, monkeypatch):
    state_file = tmp_path / "position-state.json"
    monkeypatch.setattr(api_module, "POSITION_STATE_FILE", state_file)
    monkeypatch.setattr(api_module, "LIVE_POLL_MIN_INTERVAL", 0)

    first = api_module.AudiClient()
    monkeypatch.setattr(first, "_save_live_poll_state", MagicMock())
    vehicle = FakeVehicle()
    await poll(first, vehicle)
    vehicle._mileage = 101
    await poll(first, vehicle)

    restarted = api_module.AudiClient()
    monkeypatch.setattr(restarted, "_save_live_poll_state", MagicMock())
    restored_vehicle = FakeVehicle(odometer=101, position=None)
    restarted.vehicles = [restored_vehicle]
    restarted._restore_positions()

    assert restored_vehicle._position == OLD_POSITION
    state = restarted._position_state_store.state_for(restored_vehicle.vin)
    assert state["pending_odometer"] == 101
    await poll(restarted, restored_vehicle)
    restored_vehicle._fetch_position_candidate.assert_awaited_once()


@pytest.mark.asyncio
async def test_restart_restores_newly_accepted_position(tmp_path, monkeypatch):
    state_file = tmp_path / "position-state.json"
    monkeypatch.setattr(api_module, "POSITION_STATE_FILE", state_file)
    monkeypatch.setattr(api_module, "LIVE_POLL_MIN_INTERVAL", 0)

    first = api_module.AudiClient()
    monkeypatch.setattr(first, "_save_live_poll_state", MagicMock())
    vehicle = FakeVehicle()
    await poll(first, vehicle)
    vehicle._mileage = 101
    await poll(first, vehicle)
    await poll(first, vehicle)

    restarted = api_module.AudiClient()
    restored_vehicle = FakeVehicle(odometer=101, position=None)
    restarted.vehicles = [restored_vehicle]
    restarted._restore_positions()

    assert restored_vehicle._position == {
        "lat": 48.0,
        "lon": -123.0,
        "carCapturedTimestamp": "2026-08-02T12:00:00Z",
    }
    assert restarted._position_state_store.state_for(restored_vehicle.vin)[
        "pending_odometer"
    ] is None


@pytest.mark.asyncio
async def test_restart_after_attempt_consumes_it_and_allows_only_second(tmp_path, monkeypatch):
    state_file = tmp_path / "position-state.json"
    monkeypatch.setattr(api_module, "POSITION_STATE_FILE", state_file)
    monkeypatch.setattr(api_module, "LIVE_POLL_MIN_INTERVAL", 0)

    first = api_module.AudiClient()
    monkeypatch.setattr(first, "_save_live_poll_state", MagicMock())
    vehicle = FakeVehicle()
    vehicle._fetch_position_candidate.return_value = ParkingPositionResult(
        outcome="unavailable",
        http_status=204,
    )
    await poll(first, vehicle)
    vehicle._mileage = 101
    await poll(first, vehicle)
    await poll(first, vehicle)
    vehicle._fetch_position_candidate.assert_awaited_once()

    restarted = api_module.AudiClient()
    monkeypatch.setattr(restarted, "_save_live_poll_state", MagicMock())
    restored_vehicle = FakeVehicle(odometer=101, position=None)
    restarted.vehicles = [restored_vehicle]
    restarted._restore_positions()
    await poll(restarted, restored_vehicle)
    restored_vehicle._fetch_position_candidate.assert_awaited_once()
    await poll(restarted, restored_vehicle)
    restored_vehicle._fetch_position_candidate.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("odometer", [None, "malformed", -1, 99])
async def test_missing_malformed_or_decreasing_odometer_does_not_trigger(
    position_client,
    odometer,
):
    vehicle = FakeVehicle()
    await poll(position_client, vehicle)
    vehicle._mileage = odometer
    for _ in range(2):
        await poll(position_client, vehicle)
    vehicle._fetch_position_candidate.assert_not_awaited()


@pytest.mark.asyncio
async def test_corrupt_state_disables_position_policy_but_not_selective_status(
    tmp_path,
    monkeypatch,
):
    state_file = tmp_path / "position-state.json"
    state_file.write_text("not json")
    monkeypatch.setattr(api_module, "POSITION_STATE_FILE", state_file)
    monkeypatch.setattr(api_module, "LIVE_POLL_MIN_INTERVAL", 0)
    client = api_module.AudiClient()
    monkeypatch.setattr(client, "_save_live_poll_state", MagicMock())
    vehicle = FakeVehicle()

    updated, _ = await poll(client, vehicle)

    assert updated is True
    vehicle._fetch_vehicle_data.assert_awaited_once_with(raise_on_error=True)
    vehicle._fetch_position_candidate.assert_not_awaited()
    vehicle._fetch_trip.assert_not_awaited()


def _persisted_vehicle_state(**overrides):
    state = {
        "accepted_position": OLD_POSITION,
        "accepted_position_timestamp": OLD_POSITION["carCapturedTimestamp"],
        "position_odometer": 100,
        "pending_odometer": 101,
        "pending_detected_at": "2026-08-01T12:15:00Z",
        "attempt_count": 0,
        "retry_not_before": None,
    }
    state.update(overrides)
    return {"version": 2, "vehicles": {"WAUTEST1234567890": state}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state",
    [
        _persisted_vehicle_state(pending_odometer=100),
        _persisted_vehicle_state(pending_odometer=99),
        _persisted_vehicle_state(position_odometer=None),
        _persisted_vehicle_state(attempt_count=True),
        _persisted_vehicle_state(attempt_count=1.0),
        _persisted_vehicle_state(attempt_count=3),
        _persisted_vehicle_state(
            attempt_count=0,
            retry_not_before="2026-08-01T13:00:00Z",
        ),
        _persisted_vehicle_state(
            attempt_count=2,
            retry_not_before="2026-08-01T13:00:00Z",
        ),
        _persisted_vehicle_state(pending_detected_at=None),
        _persisted_vehicle_state(
            pending_odometer=None,
            pending_detected_at=None,
            attempt_count=1,
        ),
    ],
)
async def test_semantically_impossible_state_cannot_authorize_position_request(
    tmp_path,
    monkeypatch,
    state,
):
    state_file = tmp_path / "position-state.json"
    state_file.write_text(json.dumps(state))
    monkeypatch.setattr(api_module, "POSITION_STATE_FILE", state_file)
    monkeypatch.setattr(api_module, "LIVE_POLL_MIN_INTERVAL", 0)
    client = api_module.AudiClient()
    monkeypatch.setattr(client, "_save_live_poll_state", MagicMock())
    vehicle = FakeVehicle(odometer=101)

    assert client._position_state_store.usable is False
    updated, _ = await poll(client, vehicle)

    assert updated is True
    vehicle._fetch_vehicle_data.assert_awaited_once_with(raise_on_error=True)
    vehicle._fetch_position_candidate.assert_not_awaited()


@pytest.mark.parametrize(
    ("latitude", "longitude"),
    [
        (float("nan"), 0),
        (float("inf"), 0),
        (float("-inf"), 0),
        (0, float("nan")),
        (0, float("inf")),
        (0, float("-inf")),
        (90.000001, 0),
        (-90.000001, 0),
        (0, 180.000001),
        (0, -180.000001),
        (True, 0),
        (0, False),
        ("47", 0),
        (0, None),
    ],
)
def test_invalid_coordinates_are_rejected(latitude, longitude):
    assert normalize_position(
        {
            "lat": latitude,
            "lon": longitude,
            "carCapturedTimestamp": "2026-08-01T12:00:00Z",
        }
    ) is None


@pytest.mark.parametrize(
    ("latitude", "longitude"),
    [(90, 180), (-90, -180), (0, 0), (47.5, -122.25)],
)
def test_valid_coordinate_boundaries_are_preserved(latitude, longitude):
    assert normalize_position(
        {
            "lat": latitude,
            "lon": longitude,
            "carCapturedTimestamp": "2026-08-01T12:00:00Z",
        }
    ) == {
        "lat": latitude,
        "lon": longitude,
        "carCapturedTimestamp": "2026-08-01T12:00:00Z",
    }


@pytest.mark.asyncio
async def test_persisted_state_is_versioned_normalized_and_secret_free(
    position_client,
):
    vehicle = FakeVehicle(
        position={
            **OLD_POSITION,
            "Authorization": "Bearer secret-token",
            "spin": "739185",
        }
    )
    await poll(position_client, vehicle)
    vehicle._mileage = 101
    await poll(position_client, vehicle)

    persisted_text = position_client._position_state_store.path.read_text()
    persisted = json.loads(persisted_text)

    assert persisted["version"] == 2
    assert set(persisted["vehicles"]) == {vehicle.vin}
    vehicle_state = persisted["vehicles"][vehicle.vin]
    assert set(vehicle_state) == {
        "accepted_position",
        "accepted_position_timestamp",
        "position_odometer",
        "pending_odometer",
        "pending_detected_at",
        "attempt_count",
        "retry_not_before",
    }
    accepted = vehicle_state["accepted_position"]
    assert set(accepted) == {"lat", "lon", "carCapturedTimestamp"}
    assert "secret-token" not in persisted_text
    assert "739185" not in persisted_text
