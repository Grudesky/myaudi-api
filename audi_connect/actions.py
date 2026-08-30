"""Vehicle actions - lock, unlock, climate control, preheater, charge mode."""

import asyncio
import json
import logging
import re
from collections.abc import Callable
from hashlib import sha512
from http import HTTPStatus
from typing import Optional

from aiohttp import ClientConnectionError, ClientResponseError

from .api import AudiAPI
from .endpoints import AudiEndpoints
from .exceptions import (
    ActionFailedError,
    AmbiguousActionError,
    RequestTimeoutError,
    SpinRequiredError,
)
from .utils import to_byte_array

_LOGGER = logging.getLogger(__name__)


class AudiVehicleActions:
    """Handles all write/action vehicle API calls."""

    def __init__(
        self,
        api: AudiAPI,
        endpoints: AudiEndpoints,
        bearer_token: dict,
        vw_token: dict,
        xclient_id: str,
        country: str,
        spin: Optional[str],
        api_level: int,
    ):
        self._api = api
        self._endpoints = endpoints
        self._bearer_token = bearer_token
        self._vw_token = vw_token
        self._xclient_id = xclient_id
        self._country = country
        self._type = "Audi"
        self._spin = spin
        self._api_level = api_level

    async def set_vehicle_lock(self, vin: str, lock: bool) -> None:
        """Lock or unlock the vehicle. Requires S-PIN."""
        if self._api_level == 1:
            # CARIAD path: plaintext S-PIN in the JSON body, IDK bearer token
            # (same token as climatisation). Replaces the legacy rlu_v1
            # rolesrights challenge + SHA-512 hash + XML flow, which returns
            # 403 on CARIAD EVs (e.g. Q4 e-tron) where the rlu_v1 service is
            # not provisioned. Verb path: /vehicle/v1/vehicles/{vin}/access/{lock|unlock}.
            if self._spin is None:
                raise SpinRequiredError("S-PIN is required for this action (lock/unlock)")
            headers = {
                "Authorization": "Bearer " + self._bearer_token["access_token"],
                "Content-Type": "application/json",
            }
            await self._api.request(
                "POST",
                self._endpoints.cariad_url_for_vin(
                    vin, "access/" + ("lock" if lock else "unlock")
                ),
                headers=headers, data=json.dumps({"spin": self._spin}),
            )
            return

        # Legacy MBB (api_level 0): rolesrights challenge + hashed S-PIN + XML.
        security_token = await self._get_security_token(
            vin, "rlu_v1/operations/" + ("LOCK" if lock else "UNLOCK")
        )
        headers = self._get_vehicle_action_header(
            "application/vnd.vwg.mbb.RemoteLockUnlock_v1_0_0+xml", security_token
        )
        await self._api.request(
            "POST",
            "{home}/api/bs/rlu/v1/vehicles/{vin}/{action}".format(
                home=await self._endpoints.home_region_setter(vin.upper()),
                vin=vin.upper(),
                action="lock" if lock else "unlock",
            ),
            headers=headers, data=None,
        )

    async def start_climate_control(self, vin: str, temp_c: float = 21.0) -> None:
        """Start climate control at the given temperature."""
        if self._api_level == 1:
            data = json.dumps({
                "climatisationMode": "comfort",
                "targetTemperature": int(temp_c),
                "targetTemperatureUnit": "celsius",
                "climatisationWithoutExternalPower": True,
                "climatizationAtUnlock": False,
                "windowHeatingEnabled": False,
                "zoneFrontLeftEnabled": True,
                "zoneFrontRightEnabled": True,
                "zoneRearLeftEnabled": False,
                "zoneRearRightEnabled": False,
            })
            headers = {"Authorization": "Bearer " + self._bearer_token["access_token"]}
            await self._api.request(
                "POST",
                self._endpoints.cariad_url_for_vin(vin, "climatisation/start"),
                headers=headers, data=data,
            )
        else:
            # Legacy MBB API expects temperature in deciKelvin:
            # Celsius → Kelvin: +273.15, then ×10 for deciKelvin (e.g. 21°C → 2941)
            target_temp = int(temp_c * 10 + 2731)
            data = json.dumps({
                "action": {
                    "type": "startClimatisation",
                    "settings": {
                        "targetTemperature": target_temp,
                        "climatisationWithoutHVpower": True,
                        "heaterSource": "electric",
                        "climaterElementSettings": {
                            "isClimatisationAtUnlock": False,
                            "isMirrorHeatingEnabled": False,
                        },
                    },
                }
            })
            headers = self._get_vehicle_action_header("application/json", None)
            await self._api.request(
                "POST",
                "{home}/fs-car/bs/climatisation/v1/{type}/{country}/vehicles/{vin}/climater/actions".format(
                    home=await self._endpoints.home_region(vin.upper()),
                    type=self._type, country=self._country, vin=vin.upper(),
                ),
                headers=headers, data=data,
            )

    async def stop_climate_control(self, vin: str) -> None:
        """Stop climate control."""
        if self._api_level == 1:
            headers = {"Authorization": "Bearer " + self._bearer_token["access_token"]}
            await self._api.request(
                "POST",
                self._endpoints.cariad_url_for_vin(vin, "climatisation/stop"),
                headers=headers, data=None,
            )
        else:
            data = '{"action":{"type": "stopClimatisation"}}'
            headers = self._get_vehicle_action_header("application/json", None)
            await self._api.request(
                "POST",
                "{home}/fs-car/bs/climatisation/v1/{type}/{country}/vehicles/{vin}/climater/actions".format(
                    home=await self._endpoints.home_region(vin.upper()),
                    type=self._type, country=self._country, vin=vin.upper(),
                ),
                headers=headers, data=data,
            )

    async def start_preheater(self, vin: str, duration: int = 30) -> None:
        """Start auxiliary heater for the given duration (minutes)."""
        data = json.dumps({"duration_min": duration, "spin": self._spin})
        headers = {
            "Accept": "application/json",
            "Authorization": "Bearer " + self._bearer_token["access_token"],
            "User-Agent": AudiAPI.HDR_USER_AGENT,
            "Content-Type": "application/json; charset=utf-8",
        }
        await self._api.request(
            "POST",
            self._endpoints.cariad_url_for_vin(vin, "auxiliaryheating/start"),
            headers=headers, data=data,
        )

    async def stop_preheater(self, vin: str) -> None:
        """Stop auxiliary heater."""
        headers = {
            "Accept": "application/json",
            "Authorization": "Bearer " + self._bearer_token["access_token"],
            "User-Agent": AudiAPI.HDR_USER_AGENT,
            "Content-Type": "application/json; charset=utf-8",
        }
        await self._api.request(
            "POST",
            self._endpoints.cariad_url_for_vin(vin, "auxiliaryheating/stop"),
            headers=headers, data=None,
        )

    async def start_engine(
        self,
        vin: str,
        on_submission_begin: Optional[Callable[[], None]] = None,
    ) -> str:
        """Submit a CARIAD remote engine-start command and return its request ID."""
        if self._spin is None:
            raise SpinRequiredError("S-PIN is required for remote engine start")

        headers = self._get_engine_action_headers()
        # Proof issuance has no established retry semantics. Use one application-
        # level attempt and never progress to the command POST on any failure.
        try:
            proof_response = await self._api.request_once(
                "PUT",
                self._endpoints.cariad_url(
                    "/vehicle/v1/engine/{vin}/userpromptproof",
                    vin=vin.upper(),
                ),
                headers=headers,
                data=json.dumps({"spin": self._spin}),
                allow_redirects=False,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log_engine_action_failure(
                "engine_start",
                "userpromptproof",
                "/vehicle/v1/engine/{vin}/userpromptproof",
                exc,
            )
            raise ActionFailedError(
                "Could not obtain remote engine-start authorization proof"
            ) from exc
        user_prompt_proof = (
            proof_response.get("userPromptProof")
            if isinstance(proof_response, dict)
            else None
        )
        if not user_prompt_proof:
            raise ActionFailedError(
                "Audi did not return the engine-start authorization proof"
            )

        if on_submission_begin is not None:
            # Synchronous by design: persist immediately before the first await
            # that can transmit the non-idempotent command.
            on_submission_begin()

        try:
            response = await self._api.request_once(
                "POST",
                self._endpoints.cariad_url(
                    "/vehicle/v1/engine/{vin}/start",
                    vin=vin.upper(),
                ),
                headers=headers,
                data=json.dumps(
                    {
                        "securedActivationData": user_prompt_proof,
                        "spin": self._spin,
                    }
                ),
                allow_redirects=False,
            )
            return self._engine_request_id(response, "engine-start")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # CARIAD does not document any post-submission HTTP response as proof
            # that the command was rejected before enqueueing. Fail closed.
            self._log_engine_action_failure(
                "engine_start",
                "submission",
                "/vehicle/v1/engine/{vin}/start",
                exc,
            )
            raise AmbiguousActionError(
                "Remote engine-start submission outcome is unknown"
            ) from exc

    async def stop_engine(
        self,
        vin: str,
        on_submission_begin: Optional[Callable[[], None]] = None,
    ) -> str:
        """Submit one CARIAD remote engine-stop command and return its request ID.

        Stop is deliberately not retried: its transport-level retry safety has not
        been established independently from its apparently idempotent end state.
        """
        if on_submission_begin is not None:
            on_submission_begin()

        try:
            response = await self._api.request_once(
                "POST",
                self._endpoints.cariad_url(
                    "/vehicle/v1/engine/{vin}/stop",
                    vin=vin.upper(),
                ),
                headers=self._get_engine_action_headers(),
                data=None,
                allow_redirects=False,
            )
            return self._engine_request_id(response, "engine-stop")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log_engine_action_failure(
                "engine_stop",
                "submission",
                "/vehicle/v1/engine/{vin}/stop",
                exc,
            )
            raise AmbiguousActionError(
                "Remote engine-stop submission outcome is unknown"
            ) from exc

    async def get_engine_action_status(self, vin: str, request_id: str) -> dict:
        """Look up one CARIAD pending request and map its state without polling status."""
        try:
            response = await self._api.request(
                "GET",
                self._endpoints.cariad_url_for_vin(vin, "pendingrequests"),
                headers=self._get_engine_action_headers(),
                data=None,
            )
        except (RequestTimeoutError, ConnectionError, OSError, ClientResponseError):
            return {
                "status": "unknown",
                "request_id": request_id,
                "upstream_status": None,
                "reason": "status_lookup_timeout",
            }

        pending_requests = response.get("data", []) if isinstance(response, dict) else []
        if not isinstance(pending_requests, list):
            pending_requests = []

        for pending_request in pending_requests:
            if not isinstance(pending_request, dict):
                continue
            if pending_request.get("id") != request_id:
                continue

            raw_upstream_status = pending_request.get("status")
            upstream_status = (
                raw_upstream_status
                if isinstance(raw_upstream_status, str)
                and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", raw_upstream_status)
                else None
            )
            normalized = str(upstream_status or "").lower()
            if normalized == "in_progress":
                status = "in_progress"
            elif normalized == "successful":
                status = "confirmed"
            elif normalized == "failed":
                status = "failed"
            else:
                status = "unknown"

            safe_upstream = {
                key: pending_request[key]
                for key in ("id",)
                if key in pending_request
            }
            if upstream_status is not None:
                safe_upstream["status"] = upstream_status
            return {
                "status": status,
                "request_id": request_id,
                "upstream_status": upstream_status,
                "upstream": safe_upstream,
            }

        return {
            "status": "unknown",
            "request_id": request_id,
            "upstream_status": None,
            "reason": "request_not_found",
        }

    def _get_engine_action_headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Accept-charset": "utf-8",
            "Authorization": "Bearer " + self._bearer_token["access_token"],
            "User-Agent": AudiAPI.HDR_USER_AGENT,
            "Content-Type": "application/json; charset=utf-8",
            "Accept-encoding": "gzip",
        }

    @staticmethod
    def _log_engine_action_failure(
        action: str,
        phase: str,
        endpoint: str,
        exc: Exception,
    ) -> None:
        common = (
            "Remote engine action failed action=%s phase=%s endpoint=%s "
            "exception=%s"
        )
        exception_type = type(exc).__name__

        if isinstance(exc, ClientResponseError):
            try:
                standard_reason = HTTPStatus(exc.status).phrase
            except ValueError:
                standard_reason = None
            reason = (
                exc.message
                if isinstance(exc.message, str)
                and standard_reason is not None
                and exc.message == standard_reason
                else "unavailable"
            )
            body = getattr(exc, "safe_response_body", "unavailable")
            if not isinstance(body, str) or not re.fullmatch(
                r"[A-Za-z0-9 _,=]{1,300}",
                body,
            ):
                body = "unavailable"
            _LOGGER.error(
                common
                + " category=http_response status=%s reason=%s response_body=%s",
                action,
                phase,
                endpoint,
                exception_type,
                exc.status,
                reason,
                body,
            )
            return

        if isinstance(exc, json.JSONDecodeError):
            _LOGGER.error(
                common + " category=response_parse detail=invalid_json",
                action,
                phase,
                endpoint,
                exception_type,
            )
            return

        if isinstance(exc, (RequestTimeoutError, TimeoutError)):
            category = "timeout"
        elif isinstance(
            exc,
            (ClientConnectionError, ConnectionError, OSError),
        ):
            category = "transport"
        elif isinstance(exc, AmbiguousActionError):
            category = "response_validation"
        else:
            category = "unexpected"

        _LOGGER.error(
            common + " category=%s",
            action,
            phase,
            endpoint,
            exception_type,
            category,
        )

    @staticmethod
    def _engine_request_id(response: object, action: str) -> str:
        request_id = None
        if isinstance(response, dict):
            data = response.get("data")
            if isinstance(data, dict):
                request_id = data.get("requestID")
        if not isinstance(request_id, str) or not request_id:
            raise AmbiguousActionError(
                f"Audi accepted the {action} response without a request ID"
            )
        return request_id

    # --- Security helpers ---

    async def _get_security_token(self, vin: str, action: str) -> str:
        headers = {
            "User-Agent": "okhttp/3.7.0",
            "X-App-Version": "3.14.0",
            "X-App-Name": "myAudi",
            "Accept": "application/json",
            "Authorization": "Bearer " + self._vw_token.get("access_token"),
        }
        body = await self._api.request(
            "GET",
            "{home}/api/rolesrights/authorization/v2/vehicles/{vin}/services/{action}/security-pin-auth-requested".format(
                home=await self._endpoints.home_region_setter(vin.upper()),
                vin=vin.upper(), action=action,
            ),
            headers=headers, data=None,
        )
        sec_token = body["securityPinAuthInfo"]["securityToken"]
        challenge = body["securityPinAuthInfo"]["securityPinTransmission"]["challenge"]

        pin_hash = self._generate_security_pin_hash(challenge)
        data = {
            "securityPinAuthentication": {
                "securityPin": {
                    "challenge": challenge,
                    "securityPinHash": pin_hash,
                },
                "securityToken": sec_token,
            }
        }
        headers["Content-Type"] = "application/json"
        body = await self._api.request(
            "POST",
            "{home}/api/rolesrights/authorization/v2/security-pin-auth-completed".format(
                home=await self._endpoints.home_region_setter(vin.upper())
            ),
            headers=headers, data=json.dumps(data),
        )
        return body["securityToken"]

    def _generate_security_pin_hash(self, challenge: str) -> str:
        if self._spin is None:
            raise SpinRequiredError("S-PIN is required for this action (lock/unlock)")
        pin = to_byte_array(self._spin)
        byte_challenge = to_byte_array(challenge)
        b = bytes(pin + byte_challenge)
        return sha512(b).hexdigest().upper()

    def _get_vehicle_action_header(self, content_type: str, security_token: Optional[str]) -> dict:
        headers = {
            "User-Agent": AudiAPI.HDR_USER_AGENT,
            "X-App-Version": AudiAPI.HDR_XAPP_VERSION,
            "X-App-Name": "myAudi",
            "Authorization": "Bearer " + self._vw_token.get("access_token"),
            "Accept-charset": "UTF-8",
            "Content-Type": content_type,
            "Accept": "application/json",
        }
        if security_token:
            headers["x-securityToken"] = security_token
        return headers
