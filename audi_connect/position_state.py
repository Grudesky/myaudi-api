"""Durable state for deferred, odometer-triggered parking-position refreshes."""

from __future__ import annotations

import json
import math
import os
import stat
import tempfile
import threading
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional


POSITION_STATE_VERSION = 2


class PositionStateError(Exception):
    """Raised when durable parking-position state cannot be used safely."""


def _parse_timestamp(value: object) -> Optional[datetime]:
    if not isinstance(value, str) or not value or len(value) > 64:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def normalize_position(position: object) -> Optional[dict]:
    """Retain only coordinates and the CARIAD capture timestamp."""
    if not isinstance(position, dict):
        return None
    data = position.get("data", position)
    if not isinstance(data, dict):
        return None
    latitude = data.get("lat")
    longitude = data.get("lon")
    timestamp = data.get("carCapturedTimestamp")
    if (
        isinstance(latitude, bool)
        or not isinstance(latitude, (int, float))
        or isinstance(longitude, bool)
        or not isinstance(longitude, (int, float))
        or not math.isfinite(latitude)
        or not math.isfinite(longitude)
        or not -90 <= latitude <= 90
        or not -180 <= longitude <= 180
        or _parse_timestamp(timestamp) is None
    ):
        return None
    return {
        "lat": latitude,
        "lon": longitude,
        "carCapturedTimestamp": timestamp,
    }


def newer_position(candidate: object, accepted: object) -> Optional[dict]:
    """Return a normalized candidate only when it is strictly newer."""
    normalized_candidate = normalize_position(candidate)
    if normalized_candidate is None:
        return None
    normalized_accepted = normalize_position(accepted)
    if normalized_accepted is None:
        return normalized_candidate
    if _parse_timestamp(normalized_candidate["carCapturedTimestamp"]) <= _parse_timestamp(
        normalized_accepted["carCapturedTimestamp"]
    ):
        return None
    return normalized_candidate


class PositionStateStore:
    """Atomically persist accepted positions and bounded refresh triggers."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._load_error: Optional[str] = None
        self._data: dict = {
            "version": POSITION_STATE_VERSION,
            "vehicles": {},
        }
        self._load()

    @property
    def usable(self) -> bool:
        return self._load_error is None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def state_for(self, vin: str) -> Optional[dict]:
        with self._lock:
            self._ensure_usable()
            state = self._data["vehicles"].get(vin.upper())
            return deepcopy(state) if state is not None else None

    def observe(
        self,
        vin: str,
        odometer: int,
        detected_at: str,
        current_position: object = None,
    ) -> bool:
        """Record one normal poll and return whether a bounded GET is due.

        The attempt marker is persisted before returning True, so a crash at
        the subsequent network boundary cannot cause the GET to be repeated.
        """
        if isinstance(odometer, bool) or not isinstance(odometer, int) or odometer < 0:
            return False
        if _parse_timestamp(detected_at) is None:
            raise ValueError("invalid movement-detection timestamp")
        with self._lock:
            self._ensure_usable()
            snapshot = deepcopy(self._data)
            state = self._data["vehicles"].get(vin.upper())
            if state is None:
                accepted = normalize_position(current_position)
                state = self._new_state(
                    accepted_position=accepted,
                    position_odometer=odometer,
                )
                self._data["vehicles"][vin.upper()] = state
                self._persist_or_rollback(snapshot)
                return False

            anchor = state["position_odometer"]
            pending = state["pending_odometer"]

            if pending is None:
                if anchor is None:
                    state["position_odometer"] = odometer
                elif odometer > anchor:
                    state["pending_odometer"] = odometer
                    state["pending_detected_at"] = detected_at
                    state["attempt_count"] = 0
                    state["retry_not_before"] = None
                else:
                    # Equal is idle; lower is anomalous. Neither is movement.
                    return False
                self._persist_or_rollback(snapshot)
                return False

            if odometer > pending:
                # Movement is still advancing. Coalesce without creating a GET.
                # Once both attempts were exhausted, this increase is the start
                # of a new episode and receives a fresh two-attempt budget.
                state["pending_odometer"] = odometer
                state["pending_detected_at"] = detected_at
                if state["attempt_count"] >= 2:
                    state["attempt_count"] = 0
                    state["retry_not_before"] = None
                self._persist_or_rollback(snapshot)
                return False

            if odometer < pending or state["attempt_count"] >= 2:
                return False

            retry_not_before = _parse_timestamp(state["retry_not_before"])
            current_time = _parse_timestamp(detected_at)
            if retry_not_before is not None and current_time < retry_not_before:
                return False

            # The odometer is unchanged on a later normal poll. Claim one of
            # two automatic attempts durably before crossing the GET boundary.
            state["attempt_count"] += 1
            state["retry_not_before"] = None
            self._persist_or_rollback(snapshot)
            return True

    def defer_retry(self, vin: str, retry_not_before: str) -> None:
        """Delay the remaining attempt after a valid upstream Retry-After."""
        retry_time = _parse_timestamp(retry_not_before)
        if retry_time is None:
            return
        with self._lock:
            self._ensure_usable()
            state = self._data["vehicles"].get(vin.upper())
            if (
                state is None
                or state["pending_odometer"] is None
                or state["attempt_count"] != 1
            ):
                return
            existing = _parse_timestamp(state["retry_not_before"])
            if existing is not None and existing >= retry_time:
                return
            snapshot = deepcopy(self._data)
            state["retry_not_before"] = retry_not_before
            self._persist_or_rollback(snapshot)

    def accept(self, vin: str, position: object, odometer: int) -> dict:
        """Accept a strictly newer position and clear the pending trigger."""
        candidate = normalize_position(position)
        if candidate is None:
            raise ValueError("invalid parking-position candidate")
        if isinstance(odometer, bool) or not isinstance(odometer, int) or odometer < 0:
            raise ValueError("invalid odometer")

        with self._lock:
            self._ensure_usable()
            state = self._data["vehicles"].get(vin.upper())
            if state is None:
                raise PositionStateError("parking-position state is missing")
            candidate_time = _parse_timestamp(candidate["carCapturedTimestamp"])
            accepted_time = _parse_timestamp(state["accepted_position_timestamp"])
            if accepted_time is not None and candidate_time <= accepted_time:
                raise ValueError("parking-position candidate is not newer")

            snapshot = deepcopy(self._data)
            state["accepted_position"] = candidate
            state["accepted_position_timestamp"] = candidate["carCapturedTimestamp"]
            state["position_odometer"] = odometer
            state["pending_odometer"] = None
            state["pending_detected_at"] = None
            state["attempt_count"] = 0
            state["retry_not_before"] = None
            self._persist_or_rollback(snapshot)
            return deepcopy(candidate)

    def reconcile(self, vin: str, position: object, odometer: Optional[int]) -> Optional[dict]:
        """Persist a newer position already obtained by a legacy update."""
        candidate = normalize_position(position)
        if candidate is None:
            return None
        valid_odometer = (
            isinstance(odometer, int)
            and not isinstance(odometer, bool)
            and odometer >= 0
        )

        with self._lock:
            self._ensure_usable()
            snapshot = deepcopy(self._data)
            state = self._data["vehicles"].get(vin.upper())
            if state is None:
                state = self._new_state(
                    accepted_position=candidate,
                    position_odometer=odometer if valid_odometer else None,
                )
                self._data["vehicles"][vin.upper()] = state
                self._persist_or_rollback(snapshot)
                return deepcopy(candidate)

            candidate_time = _parse_timestamp(candidate["carCapturedTimestamp"])
            accepted_time = _parse_timestamp(state["accepted_position_timestamp"])
            if accepted_time is not None and candidate_time <= accepted_time:
                return deepcopy(state["accepted_position"])

            state["accepted_position"] = candidate
            state["accepted_position_timestamp"] = candidate["carCapturedTimestamp"]
            if valid_odometer:
                state["position_odometer"] = odometer
            else:
                # A concurrent legacy status failure leaves only stale mileage.
                # The next normal poll establishes a baseline without a GET.
                state["position_odometer"] = None
            state["pending_odometer"] = None
            state["pending_detected_at"] = None
            state["attempt_count"] = 0
            state["retry_not_before"] = None
            self._persist_or_rollback(snapshot)
            return deepcopy(candidate)

    @staticmethod
    def _new_state(
        accepted_position: Optional[dict],
        position_odometer: Optional[int],
    ) -> dict:
        return {
            "accepted_position": accepted_position,
            "accepted_position_timestamp": (
                accepted_position["carCapturedTimestamp"]
                if accepted_position is not None
                else None
            ),
            "position_odometer": position_odometer,
            "pending_odometer": None,
            "pending_detected_at": None,
            "attempt_count": 0,
            "retry_not_before": None,
        }

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text())
            if not isinstance(raw, dict) or raw.get("version") != POSITION_STATE_VERSION:
                raise ValueError("unsupported parking-position state schema")
            vehicles = raw.get("vehicles")
            if not isinstance(vehicles, dict):
                raise ValueError("invalid parking-position vehicle map")
            self._validate_vehicles(vehicles)
            self._data = {"version": POSITION_STATE_VERSION, "vehicles": vehicles}
        except FileNotFoundError:
            return
        except Exception as exc:
            # Monitoring remains usable, but automatic position refresh is
            # disabled so corrupt state cannot cause repeated upstream calls.
            self._load_error = type(exc).__name__

    @classmethod
    def _validate_vehicles(cls, vehicles: dict) -> None:
        expected_fields = set(cls._new_state(None, None))
        for vin, state in vehicles.items():
            if (
                not isinstance(vin, str)
                or not vin
                or vin != vin.upper()
                or not isinstance(state, dict)
                or set(state) != expected_fields
            ):
                raise ValueError("invalid parking-position vehicle state")
            accepted = state["accepted_position"]
            if accepted is not None and normalize_position(accepted) != accepted:
                raise ValueError("invalid accepted parking position")
            accepted_timestamp = state["accepted_position_timestamp"]
            if accepted is None:
                if accepted_timestamp is not None:
                    raise ValueError("position timestamp without a position")
            elif accepted_timestamp != accepted["carCapturedTimestamp"]:
                raise ValueError("parking-position timestamp mismatch")
            for field in ("position_odometer", "pending_odometer"):
                value = state[field]
                if value is not None and (
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                ):
                    raise ValueError("invalid persisted odometer")
            pending_at = state["pending_detected_at"]
            attempt_count = state["attempt_count"]
            retry_not_before = state["retry_not_before"]
            if (
                isinstance(attempt_count, bool)
                or not isinstance(attempt_count, int)
                or attempt_count not in {0, 1, 2}
            ):
                raise ValueError("invalid parking-position attempt count")
            if state["pending_odometer"] is None:
                if (
                    pending_at is not None
                    or attempt_count != 0
                    or retry_not_before is not None
                ):
                    raise ValueError("invalid empty pending state")
            elif (
                state["position_odometer"] is None
                or state["pending_odometer"] <= state["position_odometer"]
                or _parse_timestamp(pending_at) is None
                or (
                    retry_not_before is not None
                    and (
                        attempt_count != 1
                        or _parse_timestamp(retry_not_before) is None
                    )
                )
            ):
                raise ValueError("invalid pending parking-position state")

    def _persist_or_rollback(self, snapshot: dict) -> None:
        try:
            self._persist()
        except Exception:
            self._data = snapshot
            raise

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            dir=self.path.parent,
        )
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(self._data, handle, separators=(",", ":"), sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary_name, stat.S_IRUSR | stat.S_IWUSR)
            os.replace(temporary_name, self.path)
            temporary_name = ""
            try:
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
            except OSError:
                directory_fd = None
            if directory_fd is not None:
                try:
                    try:
                        os.fsync(directory_fd)
                    except OSError:
                        pass
                finally:
                    os.close(directory_fd)
        except Exception as exc:
            try:
                if temporary_name:
                    os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise PositionStateError("Could not persist parking-position state") from exc

    def _ensure_usable(self) -> None:
        if self._load_error is not None:
            raise PositionStateError("Parking-position state is unreadable")
