import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from aioresponses import aioresponses

from audi_connect.actions import AudiVehicleActions
from audi_connect.api import AudiAPI
from audi_connect.endpoints import AudiEndpoints
from audi_connect.engine_actions import EngineActionStore
from audi_connect.exceptions import (
    ActionFailedError,
    ActionInProgressError,
    ActionNotFoundError,
    ActionPersistenceError,
    AmbiguousActionError,
    CapabilityNotSupportedError,
    InvalidActionRequestError,
    RequestTimeoutError,
    SpinRequiredError,
)
from audi_connect.logging_utils import redact
from audi_connect.models import VehicleDataResponse
from audi_connect.vehicle import AudiVehicle


def make_actions(*, country="US", spin="1234"):
    api = MagicMock()
    api.request = AsyncMock()
    api.request_once = AsyncMock()
    endpoints = AudiEndpoints(api, country=country, api_level=1)
    return AudiVehicleActions(
        api=api,
        endpoints=endpoints,
        bearer_token={"access_token": "bearer-secret"},
        vw_token={"access_token": "vw-secret"},
        xclient_id="xclient",
        country=country,
        spin=spin,
        api_level=1,
    )


def make_vehicle(
    tmp_path,
    capability_ids=("engineControl",),
    history_limit=50,
    engine_control_enabled=True,
):
    auth = MagicMock()

    async def submit_start(vin, on_submission_begin):
        on_submission_begin()
        return "request-start"

    async def submit_stop(vin, on_submission_begin):
        on_submission_begin()
        return "request-stop"

    auth.start_engine = AsyncMock(side_effect=submit_start)
    auth.stop_engine = AsyncMock(side_effect=submit_stop)
    auth.get_engine_action_status = AsyncMock()
    store = EngineActionStore(tmp_path / "engine-actions.json", history_limit)
    vehicle = AudiVehicle(
        auth,
        {"vin": "WAUTEST"},
        engine_action_store=store,
        engine_control_enabled=engine_control_enabled,
    )
    vehicle._vehicle_data = VehicleDataResponse(
        {
            "userCapabilities": {
                "capabilitiesStatus": {
                    "value": [{"id": item} for item in capability_ids]
                }
            }
        }
    )
    return vehicle, auth, store


class TestEngineStartTransport:
    @pytest.mark.asyncio
    async def test_start_uses_single_proof_then_non_retried_submission(self):
        actions = make_actions(country="US", spin="2468")
        actions._api.request_once.side_effect = [
            {"userPromptProof": "proof-secret"},
            {"data": {"requestID": "request-123"}},
        ]
        boundary = MagicMock()

        request_id = await actions.start_engine("wautest", boundary)

        assert request_id == "request-123"
        assert actions._api.request_once.await_count == 2
        proof_call, start_call = actions._api.request_once.await_args_list
        assert proof_call.args[:2] == (
            "PUT",
            "https://na.bff.cariad.digital/vehicle/v1/engine/WAUTEST/userpromptproof",
        )
        assert json.loads(proof_call.kwargs["data"]) == {"spin": "2468"}
        assert proof_call.kwargs["allow_redirects"] is False
        assert start_call.args[:2] == (
            "POST",
            "https://na.bff.cariad.digital/vehicle/v1/engine/WAUTEST/start",
        )
        assert json.loads(start_call.kwargs["data"]) == {
            "securedActivationData": "proof-secret",
            "spin": "2468",
        }
        assert start_call.kwargs["allow_redirects"] is False
        assert start_call.kwargs["headers"]["Authorization"] == "Bearer bearer-secret"
        boundary.assert_called_once_with()
        actions._api.request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_start_requires_spin_before_any_request(self):
        actions = make_actions(spin=None)
        with pytest.raises(SpinRequiredError, match="S-PIN"):
            await actions.start_engine("WAUTEST")
        actions._api.request_once.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_proof_never_reaches_start(self):
        actions = make_actions()
        actions._api.request_once.return_value = {}
        boundary = MagicMock()
        with pytest.raises(ActionFailedError, match="authorization proof"):
            await actions.start_engine("WAUTEST", boundary)
        actions._api.request_once.assert_awaited_once()
        boundary.assert_not_called()

    @pytest.mark.asyncio
    async def test_proof_failure_has_no_application_retry_or_start(self):
        actions = make_actions()
        actions._api.request_once.side_effect = ConnectionError("proof lost")
        boundary = MagicMock()
        with pytest.raises(ActionFailedError, match="authorization proof"):
            await actions.start_engine("WAUTEST", boundary)
        actions._api.request_once.assert_awaited_once()
        boundary.assert_not_called()

    @pytest.mark.asyncio
    async def test_proof_cancellation_propagates_without_start(self):
        actions = make_actions()
        actions._api.request_once.side_effect = asyncio.CancelledError
        boundary = MagicMock()
        with pytest.raises(asyncio.CancelledError):
            await actions.start_engine("WAUTEST", boundary)
        actions._api.request_once.assert_awaited_once()
        boundary.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("post_result", [{}, {"data": {}}, {"data": {"requestID": None}}])
    async def test_empty_or_missing_request_id_is_ambiguous(self, post_result):
        actions = make_actions()
        actions._api.request_once.side_effect = [
            {"userPromptProof": "proof-secret"},
            post_result,
        ]
        with pytest.raises(AmbiguousActionError, match="outcome is unknown"):
            await actions.start_engine("WAUTEST")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "post_error",
        [
            json.JSONDecodeError("bad", "{", 1),
            RequestTimeoutError("timed out"),
            ConnectionError("connection lost"),
            aiohttp.ClientResponseError(MagicMock(), (), status=500, message="error"),
        ],
    )
    async def test_every_post_failure_is_ambiguous(self, post_error):
        actions = make_actions()
        actions._api.request_once.side_effect = [
            {"userPromptProof": "proof-secret"},
            post_error,
        ]
        with pytest.raises(AmbiguousActionError, match="outcome is unknown"):
            await actions.start_engine("WAUTEST")
        assert actions._api.request_once.await_count == 2

    @pytest.mark.asyncio
    async def test_post_cancellation_propagates(self):
        actions = make_actions()
        actions._api.request_once.side_effect = [
            {"userPromptProof": "proof-secret"},
            asyncio.CancelledError,
        ]
        boundary = MagicMock()
        with pytest.raises(asyncio.CancelledError):
            await actions.start_engine("WAUTEST", boundary)
        boundary.assert_called_once_with()


class TestEngineActionDiagnostics:
    @pytest.mark.asyncio
    async def test_http_error_logs_safe_status_reason_and_body_shape(self, caplog):
        session = aiohttp.ClientSession()
        try:
            api = AudiAPI(session)
            actions = AudiVehicleActions(
                api,
                AudiEndpoints(api, country="US", api_level=1),
                {"access_token": "bearer-secret"},
                {"access_token": "vw-secret"},
                "xclient",
                "US",
                "2468",
                1,
            )
            proof_url = (
                "https://na.bff.cariad.digital/vehicle/v1/engine/"
                "WAUTEST/userpromptproof"
            )
            start_url = (
                "https://na.bff.cariad.digital/vehicle/v1/engine/"
                "WAUTEST/start"
            )
            with aioresponses() as mocked:
                mocked.put(
                    proof_url,
                    payload={"userPromptProof": "proof-secret"},
                )
                mocked.post(
                    start_url,
                    status=403,
                    reason="Forbidden",
                    payload={
                        "error": {
                            "code": "denied",
                            "message": (
                                "spin=2468 proof=proof-secret "
                                "activation=activation-secret"
                            ),
                        },
                        "Authorization": "Bearer bearer-secret",
                    },
                )
                with caplog.at_level("ERROR", logger="audi_connect.actions"):
                    with pytest.raises(AmbiguousActionError) as raised:
                        await actions.start_engine("WAUTEST")

            assert isinstance(raised.value.__cause__, aiohttp.ClientResponseError)
            log_text = caplog.text
            assert "action=engine_start" in log_text
            assert "phase=submission" in log_text
            assert "endpoint=/vehicle/v1/engine/{vin}/start" in log_text
            assert "exception=ClientResponseError" in log_text
            assert "category=http_response" in log_text
            assert "status=403" in log_text
            assert "reason=Forbidden" in log_text
            assert "response_body=json_object" in log_text
            assert "error_fields=code,message" in log_text
            for secret in (
                "2468",
                "proof-secret",
                "activation-secret",
                "bearer-secret",
            ):
                assert secret not in log_text
        finally:
            await session.close()

    @pytest.mark.asyncio
    async def test_malformed_json_logs_parse_failure_without_body(self, caplog):
        actions = make_actions(spin="2468")
        secret_body = (
            '{"spin":"2468","userPromptProof":"proof-secret",'
            '"securedActivationData":"activation-secret",'
            '"Authorization":"Bearer bearer-secret"'
        )
        parse_error = json.JSONDecodeError("malformed", secret_body, 0)
        actions._api.request_once.side_effect = [
            {"userPromptProof": "proof-secret"},
            parse_error,
        ]

        with caplog.at_level("ERROR", logger="audi_connect.actions"):
            with pytest.raises(AmbiguousActionError) as raised:
                await actions.start_engine("WAUTEST")

        assert raised.value.__cause__ is parse_error
        assert "action=engine_start" in caplog.text
        assert "phase=submission" in caplog.text
        assert "exception=JSONDecodeError" in caplog.text
        assert "category=response_parse" in caplog.text
        assert "detail=invalid_json" in caplog.text
        assert secret_body not in caplog.text
        for secret in ("2468", "proof-secret", "activation-secret", "bearer-secret"):
            assert secret not in caplog.text

    @pytest.mark.asyncio
    async def test_nonstandard_http_reason_is_not_logged(self, caplog):
        actions = make_actions(spin="2468")
        error = aiohttp.ClientResponseError(
            MagicMock(),
            (),
            status=403,
            message=(
                "Forbidden spin=2468 proof-secret "
                "activation-secret bearer-secret"
            ),
        )
        actions._api.request_once.side_effect = [
            {"userPromptProof": "proof-secret"},
            error,
        ]

        with caplog.at_level("ERROR", logger="audi_connect.actions"):
            with pytest.raises(AmbiguousActionError):
                await actions.start_engine("WAUTEST")

        assert "status=403" in caplog.text
        assert "reason=unavailable" in caplog.text
        for secret in ("2468", "proof-secret", "activation-secret", "bearer-secret"):
            assert secret not in caplog.text

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("action", "error", "category"),
        [
            (
                "engine_start",
                RequestTimeoutError("timed out with spin 2468"),
                "timeout",
            ),
            (
                "engine_start",
                aiohttp.ClientConnectionError("lost bearer-secret"),
                "transport",
            ),
            (
                "engine_stop",
                ConnectionError("lost proof-secret activation-secret"),
                "transport",
            ),
        ],
    )
    async def test_transport_categories_do_not_log_exception_messages(
        self,
        caplog,
        action,
        error,
        category,
    ):
        actions = make_actions(spin="2468")
        if action == "engine_start":
            actions._api.request_once.side_effect = [
                {"userPromptProof": "proof-secret"},
                error,
            ]
            operation = actions.start_engine("WAUTEST")
        else:
            actions._api.request_once.side_effect = error
            operation = actions.stop_engine("WAUTEST")

        with caplog.at_level("ERROR", logger="audi_connect.actions"):
            with pytest.raises(AmbiguousActionError) as raised:
                await operation

        assert raised.value.__cause__ is error
        assert f"action={action}" in caplog.text
        assert f"exception={type(error).__name__}" in caplog.text
        assert f"category={category}" in caplog.text
        for secret in ("2468", "proof-secret", "activation-secret", "bearer-secret"):
            assert secret not in caplog.text

    @pytest.mark.asyncio
    async def test_proof_failure_is_identified_without_secret_logging(self, caplog):
        actions = make_actions(spin="2468")
        error = ConnectionError("proof-secret bearer-secret 2468")
        actions._api.request_once.side_effect = error

        with caplog.at_level("ERROR", logger="audi_connect.actions"):
            with pytest.raises(ActionFailedError) as raised:
                await actions.start_engine("WAUTEST")

        assert raised.value.__cause__ is error
        assert "action=engine_start" in caplog.text
        assert "phase=userpromptproof" in caplog.text
        assert "category=transport" in caplog.text
        for secret in ("2468", "proof-secret", "bearer-secret"):
            assert secret not in caplog.text


class TestEngineStopTransport:
    @pytest.mark.asyncio
    async def test_stop_is_single_attempt_without_body_spin_or_redirects(self):
        actions = make_actions(country="DE", spin=None)
        actions._api.request_once.return_value = {"data": {"requestID": "request-stop"}}
        boundary = MagicMock()
        request_id = await actions.stop_engine("wautest", boundary)
        assert request_id == "request-stop"
        call = actions._api.request_once.await_args
        assert call.args[:2] == (
            "POST",
            "https://emea.bff.cariad.digital/vehicle/v1/engine/WAUTEST/stop",
        )
        assert call.kwargs["data"] is None
        assert call.kwargs["allow_redirects"] is False
        actions._api.request_once.assert_awaited_once()
        boundary.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_stop_transport_failure_is_ambiguous_and_not_retried(self):
        actions = make_actions()
        actions._api.request_once.side_effect = ConnectionError("connection lost")
        with pytest.raises(AmbiguousActionError, match="outcome is unknown"):
            await actions.stop_engine("WAUTEST")
        actions._api.request_once.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("redirect_status", [301, 302, 303, 307, 308])
    async def test_stop_redirect_is_ambiguous_and_not_followed(self, redirect_status):
        session = aiohttp.ClientSession()
        try:
            api = AudiAPI(session)
            actions = AudiVehicleActions(
                api,
                AudiEndpoints(api, country="US", api_level=1),
                {"access_token": "bearer-secret"},
                {"access_token": "vw-secret"},
                "xclient",
                "US",
                None,
                1,
            )
            stop_url = "https://na.bff.cariad.digital/vehicle/v1/engine/WAUTEST/stop"
            redirected_url = "https://na.bff.cariad.digital/redirected-stop"
            with aioresponses() as mocked:
                mocked.post(
                    stop_url,
                    status=redirect_status,
                    headers={"Location": redirected_url},
                )
                mocked.post(
                    redirected_url,
                    payload={"data": {"requestID": "duplicate"}},
                )
                with pytest.raises(AmbiguousActionError):
                    await actions.stop_engine("WAUTEST")
                redirected_calls = [
                    calls
                    for (method, url), calls in mocked.requests.items()
                    if method == "POST" and str(url) == redirected_url
                ]
            assert redirected_calls == []
        finally:
            await session.close()


class TestPendingRequestStatus:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("upstream_status", "expected"),
        [
            ("in_progress", "in_progress"),
            ("successful", "confirmed"),
            ("failed", "failed"),
            ("rejected", "unknown"),
            ("cancelled", "unknown"),
            ("unexpected_new_state", "unknown"),
        ],
    )
    async def test_mapping_preserves_meaningful_upstream_state(
        self, upstream_status, expected
    ):
        actions = make_actions()
        actions._api.request.return_value = {
            "data": [
                {
                    "id": "request-123",
                    "status": upstream_status,
                    "detail": "safe detail",
                    "body": {"spin": "must-not-escape"},
                }
            ]
        }
        result = await actions.get_engine_action_status("WAUTEST", "request-123")
        assert result["status"] == expected
        assert result["upstream_status"] == upstream_status
        assert "body" not in result["upstream"]

    @pytest.mark.asyncio
    async def test_missing_and_timeout_are_structured_unknown(self):
        actions = make_actions()
        actions._api.request.return_value = {"data": []}
        assert await actions.get_engine_action_status("WAUTEST", "request-123") == {
            "status": "unknown",
            "request_id": "request-123",
            "upstream_status": None,
            "reason": "request_not_found",
        }
        actions._api.request.side_effect = RequestTimeoutError("timed out")
        assert await actions.get_engine_action_status("WAUTEST", "request-123") == {
            "status": "unknown",
            "request_id": "request-123",
            "upstream_status": None,
            "reason": "status_lookup_timeout",
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("redirect_status", [301, 302, 303, 307, 308])
async def test_engine_start_never_follows_redirect(redirect_status):
    session = aiohttp.ClientSession()
    try:
        api = AudiAPI(session)
        actions = AudiVehicleActions(
            api,
            AudiEndpoints(api, country="US", api_level=1),
            {"access_token": "bearer-secret"},
            {"access_token": "vw-secret"},
            "xclient",
            "US",
            "1234",
            1,
        )
        proof_url = "https://na.bff.cariad.digital/vehicle/v1/engine/WAUTEST/userpromptproof"
        start_url = "https://na.bff.cariad.digital/vehicle/v1/engine/WAUTEST/start"
        redirected_url = "https://na.bff.cariad.digital/redirected-start"
        with aioresponses() as mocked:
            mocked.put(proof_url, payload={"userPromptProof": "proof-secret"})
            mocked.post(start_url, status=redirect_status, headers={"Location": redirected_url})
            mocked.post(redirected_url, payload={"data": {"requestID": "duplicate"}})
            with pytest.raises(AmbiguousActionError):
                await actions.start_engine("WAUTEST")
            redirected_calls = [
                calls
                for (method, url), calls in mocked.requests.items()
                if method == "POST" and str(url) == redirected_url
            ]
        assert redirected_calls == []
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_real_http_layer_does_not_retry_engine_start_post():
    session = aiohttp.ClientSession()
    try:
        api = AudiAPI(session)
        actions = AudiVehicleActions(
            api,
            AudiEndpoints(api, country="US", api_level=1),
            {"access_token": "bearer-secret"},
            {"access_token": "vw-secret"},
            "xclient",
            "US",
            "1234",
            1,
        )
        proof_url = "https://na.bff.cariad.digital/vehicle/v1/engine/WAUTEST/userpromptproof"
        start_url = "https://na.bff.cariad.digital/vehicle/v1/engine/WAUTEST/start"
        with aioresponses() as mocked:
            mocked.put(proof_url, payload={"userPromptProof": "proof-secret"})
            mocked.post(start_url, exception=ConnectionError("connection lost"))
            with pytest.raises(AmbiguousActionError):
                await actions.start_engine("WAUTEST")
            start_calls = [
                calls
                for (method, url), calls in mocked.requests.items()
                if method == "POST" and str(url) == start_url
            ]
        assert len(start_calls) == 1
        assert len(start_calls[0]) == 1
    finally:
        await session.close()


class TestDurableStateMachine:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "capability_ids",
        [(), ("engineType",), ("ignition",), ("readiness",), ("engineControl",)],
    )
    async def test_configured_start_does_not_require_capability_snapshot(
        self,
        tmp_path,
        capability_ids,
    ):
        vehicle, auth, _ = make_vehicle(tmp_path, capability_ids)
        assert await vehicle.start_engine() == "request-start"
        auth.start_engine.assert_awaited_once()
        assert vehicle.engine_control_capability_advertised is (
            "engineControl" in capability_ids
        )
        auth.get_stored_vehicle_data.assert_not_called()

    @pytest.mark.asyncio
    async def test_unconfigured_vehicle_fails_closed_before_submission(self, tmp_path):
        vehicle, auth, _ = make_vehicle(
            tmp_path,
            capability_ids=("engineControl",),
            engine_control_enabled=False,
        )
        with pytest.raises(CapabilityNotSupportedError, match="configuration"):
            await vehicle.start_engine()
        auth.start_engine.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_durable_store_is_required(self):
        auth = MagicMock()
        vehicle = AudiVehicle(
            auth,
            {"vin": "WAUTEST"},
            engine_control_enabled=True,
        )
        vehicle._vehicle_data = VehicleDataResponse(
            {"userCapabilities": {"capabilitiesStatus": {"value": [{"id": "engineControl"}]}}}
        )
        with pytest.raises(ActionPersistenceError, match="Durable"):
            await vehicle.start_engine()

    @pytest.mark.asyncio
    async def test_sent_request_id_is_retained(self, tmp_path):
        vehicle, _, store = make_vehicle(tmp_path)
        assert await vehicle.start_engine() == "request-start"
        active = store.active_for("WAUTEST")
        assert active["status"] == "sent"
        assert active["request_id"] == "request-start"

    @pytest.mark.asyncio
    async def test_concurrent_duplicate_is_blocked(self, tmp_path):
        vehicle, auth, _ = make_vehicle(tmp_path)
        entered = asyncio.Event()
        release = asyncio.Event()

        async def slow_start(vin, boundary):
            boundary()
            entered.set()
            await release.wait()
            return "request-start"

        auth.start_engine.side_effect = slow_start
        first = asyncio.create_task(vehicle.start_engine())
        await entered.wait()
        second = asyncio.create_task(vehicle.start_engine())
        await asyncio.sleep(0)
        release.set()
        assert await first == "request-start"
        with pytest.raises(ActionInProgressError):
            await second
        auth.start_engine.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cancellation_before_post_clears_marker_and_propagates(self, tmp_path):
        vehicle, auth, store = make_vehicle(tmp_path)
        auth.start_engine.side_effect = asyncio.CancelledError
        with pytest.raises(asyncio.CancelledError):
            await vehicle.start_engine()
        assert store.active_for("WAUTEST") is None

    @pytest.mark.asyncio
    async def test_cancellation_during_post_remains_ambiguous_after_restart(self, tmp_path):
        vehicle, auth, store = make_vehicle(tmp_path)

        async def cancel_during_post(vin, boundary):
            boundary()
            raise asyncio.CancelledError

        auth.start_engine.side_effect = cancel_during_post
        with pytest.raises(asyncio.CancelledError):
            await vehicle.start_engine()
        assert store.active_for("WAUTEST")["status"] == "ambiguous"
        restarted = AudiVehicle(
            auth,
            {"vin": "WAUTEST"},
            engine_action_store=EngineActionStore(store.path),
            engine_control_enabled=True,
        )
        restarted._vehicle_data = vehicle._vehicle_data
        with pytest.raises(ActionInProgressError):
            await restarted.start_engine()

    @pytest.mark.asyncio
    async def test_ambiguous_exception_remains_durable(self, tmp_path):
        vehicle, auth, store = make_vehicle(tmp_path)

        async def ambiguous(vin, boundary):
            boundary()
            raise AmbiguousActionError("unknown")

        auth.start_engine.side_effect = ambiguous
        with pytest.raises(AmbiguousActionError):
            await vehicle.start_engine()
        assert store.active_for("WAUTEST")["status"] == "ambiguous"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "post_result",
        [
            json.JSONDecodeError("bad", "{", 1),
            {},
            {"data": {}},
            RequestTimeoutError("timed out"),
            ConnectionError("connection lost"),
            aiohttp.ClientResponseError(MagicMock(), (), status=500, message="error"),
        ],
    )
    async def test_every_uncertain_start_response_keeps_durable_guard(
        self, tmp_path, post_result
    ):
        vehicle, auth, store = make_vehicle(tmp_path)
        actions = make_actions()
        actions._api.request_once.side_effect = [
            {"userPromptProof": "proof-secret"},
            post_result,
        ]
        auth.start_engine.side_effect = actions.start_engine

        with pytest.raises(AmbiguousActionError):
            await vehicle.start_engine()

        assert store.active_for("WAUTEST")["status"] == "ambiguous"
        restarted = AudiVehicle(
            auth,
            {"vin": "WAUTEST"},
            engine_action_store=EngineActionStore(store.path),
            engine_control_enabled=True,
        )
        restarted._vehicle_data = vehicle._vehicle_data
        with pytest.raises(ActionInProgressError):
            await restarted.start_engine()

    @pytest.mark.asyncio
    async def test_ambiguous_stop_is_durable_and_blocks_restart(self, tmp_path):
        vehicle, auth, store = make_vehicle(tmp_path)

        async def ambiguous_stop(vin, boundary):
            boundary()
            raise AmbiguousActionError("unknown")

        auth.stop_engine.side_effect = ambiguous_stop
        with pytest.raises(AmbiguousActionError):
            await vehicle.stop_engine()
        assert store.active_for("WAUTEST")["action"] == "engine_stop"
        restarted = AudiVehicle(
            auth,
            {"vin": "WAUTEST"},
            engine_action_store=EngineActionStore(store.path),
            engine_control_enabled=True,
        )
        restarted._vehicle_data = vehicle._vehicle_data
        with pytest.raises(ActionInProgressError):
            await restarted.stop_engine()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("action", ["engine_start", "engine_stop"])
    async def test_diagnostic_logging_preserves_durable_ambiguous_state(
        self,
        tmp_path,
        caplog,
        action,
    ):
        vehicle, auth, store = make_vehicle(tmp_path)
        actions = make_actions()
        error = ConnectionError("transport detail with proof-secret 1234")
        if action == "engine_start":
            actions._api.request_once.side_effect = [
                {"userPromptProof": "proof-secret"},
                error,
            ]
            auth.start_engine.side_effect = actions.start_engine
            operation = vehicle.start_engine()
        else:
            actions._api.request_once.side_effect = error
            auth.stop_engine.side_effect = actions.stop_engine
            operation = vehicle.stop_engine()

        with caplog.at_level("ERROR", logger="audi_connect.actions"):
            with pytest.raises(AmbiguousActionError):
                await operation

        active = store.active_for("WAUTEST")
        assert active["action"] == action
        assert active["status"] == "ambiguous"
        assert active["request_id"] is None
        assert f"action={action}" in caplog.text
        assert "category=transport" in caplog.text
        assert "proof-secret" not in caplog.text
        assert "1234" not in caplog.text

    @pytest.mark.asyncio
    async def test_restart_with_submitting_blocks_start(self, tmp_path):
        store = EngineActionStore(tmp_path / "engine-actions.json")
        store.begin("WAUTEST", "engine_start")
        vehicle, auth, _ = make_vehicle(tmp_path)
        with pytest.raises(ActionInProgressError):
            await vehicle.start_engine()
        auth.start_engine.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", ["sent", "in_progress", "ambiguous"])
    async def test_restart_with_unresolved_state_blocks_start(self, tmp_path, status):
        store = EngineActionStore(tmp_path / "engine-actions.json")
        record = store.begin("WAUTEST", "engine_start")
        request_id = None if status == "ambiguous" else "request-start"
        store.transition("WAUTEST", record["local_action_id"], status, request_id)
        vehicle, auth, _ = make_vehicle(tmp_path)
        with pytest.raises(ActionInProgressError):
            await vehicle.start_engine()
        auth.start_engine.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("result", "stored_status"),
        [
            (
                {
                    "status": "in_progress",
                    "request_id": "request-start",
                    "upstream_status": "in_progress",
                },
                "in_progress",
            ),
            (
                {
                    "status": "unknown",
                    "request_id": "request-start",
                    "upstream_status": "new_state",
                },
                "ambiguous",
            ),
            (
                {
                    "status": "unknown",
                    "request_id": "request-start",
                    "upstream_status": None,
                    "reason": "request_not_found",
                },
                "ambiguous",
            ),
            (
                {
                    "status": "unknown",
                    "request_id": "request-start",
                    "upstream_status": None,
                    "reason": "status_lookup_timeout",
                },
                "ambiguous",
            ),
        ],
    )
    async def test_nonterminal_lookup_never_clears_guard(self, tmp_path, result, stored_status):
        vehicle, auth, store = make_vehicle(tmp_path)
        await vehicle.start_engine()
        auth.get_engine_action_status.return_value = result
        assert await vehicle.get_engine_action_status("request-start") == result
        assert store.active_for("WAUTEST")["status"] == stored_status
        with pytest.raises(ActionInProgressError):
            await vehicle.start_engine()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("terminal", ["confirmed", "failed"])
    async def test_only_positive_terminal_result_auto_clears(self, tmp_path, terminal):
        vehicle, auth, store = make_vehicle(tmp_path)
        await vehicle.start_engine()
        auth.get_engine_action_status.return_value = {
            "status": terminal,
            "request_id": "request-start",
            "upstream_status": "successful" if terminal == "confirmed" else "failed",
        }
        await vehicle.get_engine_action_status("request-start")
        assert store.active_for("WAUTEST") is None
        assert await vehicle.stop_engine() == "request-stop"

    @pytest.mark.asyncio
    async def test_explicit_recovery_clears_ambiguous_without_vehicle_call(self, tmp_path):
        vehicle, auth, store = make_vehicle(tmp_path)

        async def ambiguous(vin, boundary):
            boundary()
            raise AmbiguousActionError("unknown")

        auth.start_engine.side_effect = ambiguous
        with pytest.raises(AmbiguousActionError):
            await vehicle.start_engine()
        local_id = store.active_for("WAUTEST")["local_action_id"]
        auth.reset_mock()
        recovered = await vehicle.recover_engine_action(local_id)
        assert recovered["status"] == "operator_recovered"
        assert store.active_for("WAUTEST") is None
        auth.start_engine.assert_not_awaited()
        auth.stop_engine.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_invalid_and_arbitrary_request_ids_never_query_upstream(self, tmp_path):
        vehicle, auth, _ = make_vehicle(tmp_path)
        with pytest.raises(InvalidActionRequestError):
            await vehicle.get_engine_action_status("../bad")
        with pytest.raises(ActionNotFoundError):
            await vehicle.get_engine_action_status("unknown-request")
        auth.get_engine_action_status.assert_not_awaited()

    def test_terminal_history_is_bounded(self, tmp_path):
        store = EngineActionStore(tmp_path / "engine-actions.json", history_limit=2)
        for index in range(3):
            record = store.begin("WAUTEST", "engine_stop")
            store.transition(
                "WAUTEST", record["local_action_id"], "sent", f"request-{index}"
            )
            store.resolve_terminal("WAUTEST", record["local_action_id"], "confirmed")
        assert [item["request_id"] for item in store.history_for("WAUTEST")] == [
            "request-1",
            "request-2",
        ]

    def test_persisted_state_contains_no_engine_secrets(self, tmp_path):
        store = EngineActionStore(tmp_path / "engine-actions.json")
        record = store.begin("WAUTEST", "engine_start")
        store.transition(
            "WAUTEST", record["local_action_id"], "sent", "request-start"
        )
        serialized = store.path.read_text()
        for secret in (
            "1234",
            "bearer-secret",
            "proof-secret",
            "activation-secret",
            "spin",
            "Authorization",
            "userPromptProof",
            "securedActivationData",
        ):
            assert secret not in serialized

    @pytest.mark.asyncio
    async def test_corrupt_persisted_state_disables_commands_without_status_refresh(
        self, tmp_path
    ):
        state_file = tmp_path / "engine-actions.json"
        state_file.write_text('{"version":1,"vehicles":{"WAUTEST":{"active":{"spin":"1234"}}}}')
        store = EngineActionStore(state_file)
        auth = MagicMock()
        vehicle = AudiVehicle(
            auth,
            {"vin": "WAUTEST"},
            engine_action_store=store,
            engine_control_enabled=True,
        )
        vehicle._vehicle_data = VehicleDataResponse(
            {"userCapabilities": {"capabilitiesStatus": {"value": [{"id": "engineControl"}]}}}
        )
        with pytest.raises(ActionPersistenceError, match="unreadable"):
            await vehicle.start_engine()
        auth.start_engine.assert_not_called()
        auth.get_stored_vehicle_data.assert_not_called()


def test_engine_secrets_are_redacted():
    value = (
        "Authorization: Bearer bearer-secret "
        '{"spin":"2468","userPromptProof":"proof-secret",'
        '"securedActivationData":"activation-secret"}'
    )
    redacted = redact(value)
    for secret in ("bearer-secret", "2468", "proof-secret", "activation-secret"):
        assert secret not in redacted
