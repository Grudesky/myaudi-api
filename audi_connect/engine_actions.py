"""Durable, secret-free state for safety-critical engine commands."""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .exceptions import (
    ActionInProgressError,
    ActionNotFoundError,
    ActionPersistenceError,
    InvalidActionRequestError,
)

ENGINE_ACTION_HISTORY_LIMIT = 50
ENGINE_ACTION_STATE_VERSION = 1
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def default_engine_action_state_file() -> Path:
    return Path(
        os.getenv(
            "AUDI_ENGINE_ACTION_STATE_FILE",
            str(Path.home() / ".myaudi-api" / "engine-actions.json"),
        )
    )


def validate_request_id(request_id: str) -> str:
    if not isinstance(request_id, str) or not REQUEST_ID_PATTERN.fullmatch(request_id):
        raise InvalidActionRequestError("Invalid CARIAD request ID")
    return request_id


class EngineActionStore:
    """Atomically persist unresolved actions and a bounded terminal history.

    The serialized schema is deliberately fixed. Callers cannot add payloads,
    headers, proofs, tokens, S-PINs, or arbitrary upstream response data.
    """

    def __init__(
        self,
        path: Path | str,
        history_limit: int = ENGINE_ACTION_HISTORY_LIMIT,
    ) -> None:
        self.path = Path(path)
        self.history_limit = max(1, int(history_limit))
        self._lock = threading.RLock()
        self._load_error: Optional[str] = None
        self._data: dict = {
            "version": ENGINE_ACTION_STATE_VERSION,
            "vehicles": {},
        }
        self._load()

    def active_for(self, vin: str) -> Optional[dict]:
        with self._lock:
            self._ensure_usable()
            active = self._vehicle(vin).get("active")
            return deepcopy(active) if active else None

    def history_for(self, vin: str) -> list[dict]:
        with self._lock:
            self._ensure_usable()
            return deepcopy(self._vehicle(vin).get("history", []))

    def begin(self, vin: str, action: str) -> dict:
        if action not in {"engine_start", "engine_stop"}:
            raise InvalidActionRequestError("Invalid engine action")
        now = self._now()
        with self._lock:
            self._ensure_usable()
            snapshot = deepcopy(self._data)
            vehicle = self._vehicle(vin)
            if vehicle.get("active") is not None:
                raise ActionInProgressError(
                    "Another engine action is still unresolved for this vehicle"
                )
            record = {
                "local_action_id": str(uuid.uuid4()),
                "vin": vin.upper(),
                "action": action,
                "status": "submitting",
                "request_id": None,
                "created_at": now,
                "updated_at": now,
            }
            vehicle["active"] = record
            try:
                self._persist()
            except Exception:
                self._data = snapshot
                raise
            return deepcopy(record)

    def transition(
        self,
        vin: str,
        local_action_id: str,
        status: str,
        request_id: Optional[str] = None,
    ) -> dict:
        if status not in {"ambiguous", "sent", "in_progress"}:
            raise InvalidActionRequestError("Invalid unresolved action state")
        if request_id is not None:
            validate_request_id(request_id)
        with self._lock:
            snapshot = deepcopy(self._data)
            active = self._matching_active(vin, local_action_id)
            active["status"] = status
            if request_id is not None:
                active["request_id"] = request_id
            active["updated_at"] = self._now()
            try:
                self._persist()
            except Exception:
                self._data = snapshot
                raise
            return deepcopy(active)

    def abort_before_submission(self, vin: str, local_action_id: str) -> None:
        """Clear only a marker that never crossed the command POST boundary."""
        with self._lock:
            snapshot = deepcopy(self._data)
            active = self._matching_active(vin, local_action_id)
            if active.get("status") != "submitting":
                raise ActionPersistenceError(
                    "Refusing to clear an action after command submission began"
                )
            self._vehicle(vin)["active"] = None
            try:
                self._persist()
            except Exception:
                self._data = snapshot
                raise

    def resolve_terminal(
        self,
        vin: str,
        local_action_id: str,
        status: str,
    ) -> dict:
        if status not in {"confirmed", "failed"}:
            raise InvalidActionRequestError("Invalid terminal action state")
        with self._lock:
            snapshot = deepcopy(self._data)
            active = self._matching_active(vin, local_action_id)
            terminal = {**active, "status": status, "updated_at": self._now()}
            self._append_history(self._vehicle(vin), terminal)
            self._vehicle(vin)["active"] = None
            try:
                self._persist()
            except Exception:
                self._data = snapshot
                raise
            return deepcopy(terminal)

    def recover(self, vin: str, local_action_id: str) -> dict:
        """Explicitly clear one unresolved action without issuing a vehicle command."""
        with self._lock:
            snapshot = deepcopy(self._data)
            active = self._matching_active(vin, local_action_id)
            recovered = {
                **active,
                "status": "operator_recovered",
                "updated_at": self._now(),
            }
            self._append_history(self._vehicle(vin), recovered)
            self._vehicle(vin)["active"] = None
            try:
                self._persist()
            except Exception:
                self._data = snapshot
                raise
            return deepcopy(recovered)

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text())
            if not isinstance(raw, dict) or raw.get("version") != ENGINE_ACTION_STATE_VERSION:
                raise ValueError("unsupported engine-action state schema")
            vehicles = raw.get("vehicles")
            if not isinstance(vehicles, dict):
                raise ValueError("invalid engine-action vehicle map")
            self._validate_loaded_vehicles(vehicles)
            self._data = {"version": ENGINE_ACTION_STATE_VERSION, "vehicles": vehicles}
        except FileNotFoundError:
            return
        except Exception as exc:
            # Corrupt or unreadable durable state must fail closed. Do not replace it.
            self._load_error = type(exc).__name__

    def _vehicle(self, vin: str) -> dict:
        vehicles = self._data["vehicles"]
        return vehicles.setdefault(vin.upper(), {"active": None, "history": []})

    def _matching_active(self, vin: str, local_action_id: str) -> dict:
        self._ensure_usable()
        active = self._vehicle(vin).get("active")
        if not active or active.get("local_action_id") != local_action_id:
            raise ActionNotFoundError("Engine action was not found for this vehicle")
        return active

    def _append_history(self, vehicle: dict, record: dict) -> None:
        history = vehicle.setdefault("history", [])
        history.append(record)
        del history[:-self.history_limit]

    def _validate_loaded_vehicles(self, vehicles: dict) -> None:
        allowed_fields = {
            "local_action_id",
            "vin",
            "action",
            "status",
            "request_id",
            "created_at",
            "updated_at",
        }
        allowed_statuses = {
            "submitting",
            "ambiguous",
            "sent",
            "in_progress",
            "confirmed",
            "failed",
            "operator_recovered",
        }
        for vin, vehicle in vehicles.items():
            if not isinstance(vin, str) or not isinstance(vehicle, dict):
                raise ValueError("invalid engine-action vehicle record")
            if set(vehicle) - {"active", "history"}:
                raise ValueError("unexpected engine-action vehicle fields")
            active = vehicle.get("active")
            history = vehicle.get("history", [])
            if active is not None:
                self._validate_loaded_record(
                    active,
                    vin,
                    allowed_fields,
                    allowed_statuses
                    - {"confirmed", "failed", "operator_recovered"},
                )
            if not isinstance(history, list):
                raise ValueError("invalid engine-action history")
            for record in history[-self.history_limit :]:
                self._validate_loaded_record(
                    record, vin, allowed_fields, {"confirmed", "failed", "operator_recovered"}
                )
            vehicle["history"] = history[-self.history_limit :]

    @staticmethod
    def _validate_loaded_record(
        record: object,
        vin: str,
        allowed_fields: set[str],
        allowed_statuses: set[str],
    ) -> None:
        if not isinstance(record, dict) or set(record) != allowed_fields:
            raise ValueError("invalid engine-action record schema")
        uuid.UUID(record["local_action_id"])
        if record["vin"] != vin.upper():
            raise ValueError("engine-action VIN mismatch")
        if record["action"] not in {"engine_start", "engine_stop"}:
            raise ValueError("invalid persisted engine action")
        if record["status"] not in allowed_statuses:
            raise ValueError("invalid persisted engine-action state")
        if record["request_id"] is not None:
            validate_request_id(record["request_id"])
        if not all(
            isinstance(record[field], str) and len(record[field]) <= 64
            for field in ("created_at", "updated_at")
        ):
            raise ValueError("invalid persisted engine-action timestamp")

    def _persist(self) -> None:
        self._ensure_usable()
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
                        # The replacement succeeded. Some filesystems cannot
                        # fsync directories; do not roll memory behind disk.
                        pass
                finally:
                    os.close(directory_fd)
        except Exception as exc:
            try:
                if temporary_name:
                    os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise ActionPersistenceError("Could not persist engine-action state") from exc

    def _ensure_usable(self) -> None:
        if self._load_error is not None:
            raise ActionPersistenceError(
                "Engine-action state is unreadable; commands are disabled"
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
