"""Vehicle data client - fetches vehicle status, position, and trip data from Audi/VW APIs."""

import json
import logging
from dataclasses import dataclass
from datetime import timedelta, datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

from .api import AudiAPI
from .endpoints import AudiEndpoints
from .exceptions import AuthenticationError

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParkingPositionResult:
    """Structured result for a single candidate parking-position request."""

    outcome: str
    position: Optional[dict] = None
    http_status: Optional[int] = None
    error_type: Optional[str] = None
    retry_not_before: Optional[str] = None


def _retry_not_before(value: Optional[str]) -> Optional[str]:
    """Convert an HTTP Retry-After value to an absolute UTC timestamp."""
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    now = datetime.now(timezone.utc)
    try:
        seconds = int(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return retry_at.astimezone(timezone.utc).isoformat()
    if seconds < 0:
        return None
    return (now + timedelta(seconds=seconds)).isoformat()


class AudiVehicleClient:
    """Handles all read-only vehicle data API calls."""

    def __init__(
        self,
        api: AudiAPI,
        endpoints: AudiEndpoints,
        bearer_token: dict,
        vw_token: dict,
        audi_token: dict,
        xclient_id: str,
        country: str,
        language: str,
        api_level: int,
    ):
        self._api = api
        self._endpoints = endpoints
        self._bearer_token = bearer_token
        self._vw_token = vw_token
        self._audi_token = audi_token
        self._xclient_id = xclient_id
        self._country = country
        self._language = language
        self._type = "Audi"
        self._api_level = api_level

    async def get_vehicle_list(self) -> list[dict]:
        """Fetch the list of vehicles from the GraphQL API."""
        headers = {
            "Accept": "application/json",
            "Accept-Charset": "utf-8",
            "X-App-Name": "myAudi",
            "X-App-Version": AudiAPI.HDR_XAPP_VERSION,
            "Accept-Language": f"{self._language}-{self._country.upper()}",
            "X-User-Country": self._country.upper(),
            "User-Agent": AudiAPI.HDR_USER_AGENT,
            "Authorization": "Bearer " + self._audi_token["access_token"],
            "Content-Type": "application/json; charset=utf-8",
        }
        graphql_query = {
            "query": (
                "query vehicleList {\n userVehicles {\n vin\n mappingVin\n "
                "vehicle { core { modelYear\n }\n media { shortName\n longName }\n }\n "
                "csid\n commissionNumber\n type\n devicePlatform\n mbbConnect\n "
                "userRole {\n role\n }\n vehicle {\n classification {\n driveTrain\n }\n }\n "
                "nickname\n }\n}"
            )
        }
        graphql_url = (
            "https://app-api.my.aoa.audi.com/vgql/v1/graphql"
            if self._country.upper() == "US"
            else "https://app-api.live-my.audi.com/vgql/v1/graphql"
        )

        _, rsptxt = await self._api.request(
            "POST", graphql_url, json.dumps(graphql_query),
            headers=headers, allow_redirects=False, rsp_wtxt=True,
        )
        vins = json.loads(rsptxt)

        if "errors" in vins:
            raise AuthenticationError(f"GraphQL API returned errors: {vins['errors']}")
        if "data" not in vins or vins["data"] is None:
            raise AuthenticationError("No data in API response")
        if vins["data"].get("userVehicles") is None:
            raise AuthenticationError("No vehicle data - possible authentication issue")

        return vins["data"]["userVehicles"]

    async def get_stored_vehicle_data(self, vin: str) -> dict:
        """Fetch selective vehicle status data from CARIAD API."""
        jobs = {
            "access", "activeVentilation", "auxiliaryHeating", "batteryChargingCare",
            "batterySupport", "charging", "chargingProfiles", "chargingTimers",
            "climatisation", "climatisationTimers", "departureProfiles",
            "departureTimers", "fuelStatus", "honkAndFlash",
            "hybridCarAuxiliaryHeating", "lvBattery", "measurements", "oilLevel",
            "readiness", "vehicleHealthInspection", "vehicleHealthWarnings",
            "vehicleLights", "userCapabilities",
        }
        self._api.use_token(self._bearer_token)
        return await self._api.get(
            self._endpoints.cariad_url_for_vin(
                vin, "selectivestatus?jobs={jobs}", jobs=",".join(jobs)
            )
        )

    async def get_stored_position(self, vin: str) -> Optional[dict]:
        """Fetch vehicle parking position."""
        self._api.use_token(self._bearer_token)
        try:
            return await self._api.get(
                self._endpoints.cariad_url_for_vin(vin, "parkingposition")
            )
        except Exception:
            return None

    async def get_parking_position_candidate(self, vin: str) -> ParkingPositionResult:
        """Fetch one candidate while preserving meaningful HTTP outcomes.

        This deliberately uses the non-retrying transport entry point. The
        deferred state machine permits one automatic request per movement.
        """
        self._api.use_token(self._bearer_token)
        try:
            response, body = await self._api.get_single_transmission(
                self._endpoints.cariad_url_for_vin(vin, "parkingposition"),
                allow_redirects=False,
                rsp_wtxt=True,
            )
        except Exception as exc:
            return ParkingPositionResult(
                outcome="transport_error",
                error_type=type(exc).__name__,
            )

        status = response.status
        if status == 200:
            try:
                position = json.loads(body)
            except (json.JSONDecodeError, TypeError):
                return ParkingPositionResult(
                    outcome="invalid_response",
                    http_status=status,
                )
            if not isinstance(position, dict):
                return ParkingPositionResult(
                    outcome="invalid_response",
                    http_status=status,
                )
            return ParkingPositionResult(
                outcome="success",
                position=position,
                http_status=status,
            )
        if status == 204:
            return ParkingPositionResult(outcome="unavailable", http_status=status)
        if status == 404:
            return ParkingPositionResult(outcome="unsupported", http_status=status)
        if status == 429:
            return ParkingPositionResult(
                outcome="rate_limited",
                http_status=status,
                retry_not_before=_retry_not_before(response.headers.get("Retry-After")),
            )
        if status >= 500:
            return ParkingPositionResult(outcome="server_error", http_status=status)
        return ParkingPositionResult(outcome="http_error", http_status=status)

    async def get_tripdata(self, vin: str, kind: str) -> dict:
        """Fetch trip statistics (short-term or long-term)."""
        self._api.use_token(self._vw_token)
        headers = {
            "Accept": "application/json",
            "Accept-Charset": "utf-8",
            "X-App-Name": "myAudi",
            "X-App-Version": AudiAPI.HDR_XAPP_VERSION,
            "X-Client-ID": self._xclient_id,
            "User-Agent": AudiAPI.HDR_USER_AGENT,
            "Authorization": "Bearer " + self._vw_token["access_token"],
        }
        td_reqdata = {
            "type": "list",
            "from": "1970-01-01T00:00:00Z",
            "to": (datetime.now(timezone.utc) + timedelta(minutes=90)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        }
        return await self._api.request(
            "GET",
            "{home}/api/bs/tripstatistics/v1/vehicles/{vin}/tripdata/{kind}".format(
                home=await self._endpoints.home_region_setter(vin.upper()),
                vin=vin.upper(), kind=kind,
            ),
            None, params=td_reqdata, headers=headers,
        )
