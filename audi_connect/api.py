"""Low-level HTTP client for Audi Connect API calls."""

import json
import logging
import asyncio
from typing import Any, Optional, Union
from asyncio import TimeoutError
from aiohttp import ClientSession, ClientResponse, ClientResponseError
from aiohttp.hdrs import METH_GET

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .exceptions import RequestTimeoutError

TIMEOUT = 30
MAX_RETRIES = 3
_LOGGER = logging.getLogger(__name__)

_SAFE_ERROR_BODY_FIELDS = frozenset(
    {
        "code",
        "detail",
        "error",
        "errorCode",
        "message",
        "reason",
        "status",
    }
)


def _safe_error_body_summary(raw_body: str) -> str:
    """Describe an HTTP error body without retaining or logging its values."""
    size = len(raw_body.encode("utf-8", errors="replace"))
    try:
        parsed = json.loads(raw_body)
    except (json.JSONDecodeError, TypeError):
        return f"non_json bytes={size}"

    if not isinstance(parsed, dict):
        return f"json_{type(parsed).__name__} bytes={size}"

    fields = sorted(set(parsed).intersection(_SAFE_ERROR_BODY_FIELDS))
    error_fields: list[str] = []
    if isinstance(parsed.get("error"), dict):
        error_fields = sorted(
            set(parsed["error"]).intersection(_SAFE_ERROR_BODY_FIELDS)
        )

    field_text = ",".join(fields) or "none"
    result = f"json_object fields={field_text} bytes={size}"
    if error_fields:
        result += f" error_fields={','.join(error_fields)}"
    return result


class AudiAPI:
    HDR_XAPP_VERSION: str = "4.31.0"
    HDR_USER_AGENT: str = (
        "Android/4.31.0 (Build 800341641.root project "
        "'myaudi_android'.ext.buildTime) Android/13"
    )

    def __init__(self, session: ClientSession, proxy: Optional[str] = None):
        self._token: Optional[dict] = None
        self._xclient_id: Optional[str] = None
        self._session = session
        self._proxy: Optional[dict] = {"http": proxy, "https": proxy} if proxy else None

    def use_token(self, token: Optional[dict]) -> None:
        self._token = token

    def set_xclient_id(self, xclient_id: Optional[str]) -> None:
        self._xclient_id = xclient_id

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((RequestTimeoutError, ConnectionError, OSError)),
        reraise=True,
    )
    async def request(
        self,
        method: str,
        url: str,
        data: Any,
        headers: Optional[dict] = None,
        raw_reply: bool = False,
        raw_contents: bool = False,
        rsp_wtxt: bool = False,
        **kwargs: Any,
    ) -> Union[dict, bytes, ClientResponse, tuple[ClientResponse, str]]:
        return await self._request_once(
            method,
            url,
            data,
            headers=headers,
            raw_reply=raw_reply,
            raw_contents=raw_contents,
            rsp_wtxt=rsp_wtxt,
            **kwargs,
        )

    async def request_once(
        self,
        method: str,
        url: str,
        data: Any,
        headers: Optional[dict] = None,
        raw_reply: bool = False,
        raw_contents: bool = False,
        rsp_wtxt: bool = False,
        **kwargs: Any,
    ) -> Union[dict, bytes, ClientResponse, tuple[ClientResponse, str]]:
        """Perform exactly one HTTP attempt for non-idempotent actions."""
        return await self._request_once(
            method,
            url,
            data,
            headers=headers,
            raw_reply=raw_reply,
            raw_contents=raw_contents,
            rsp_wtxt=rsp_wtxt,
            **kwargs,
        )

    async def _request_once(
        self,
        method: str,
        url: str,
        data: Any,
        headers: Optional[dict] = None,
        raw_reply: bool = False,
        raw_contents: bool = False,
        rsp_wtxt: bool = False,
        **kwargs: Any,
    ) -> Union[dict, bytes, ClientResponse, tuple[ClientResponse, str]]:
        try:
            async with asyncio.timeout(TIMEOUT):
                async with self._session.request(
                    method, url, headers=headers, data=data, **kwargs
                ) as response:
                    if raw_reply:
                        return response

                    if rsp_wtxt:
                        txt = await response.text()
                        return response, txt

                    elif raw_contents:
                        return await response.read()

                    elif response.status in (200, 202, 207):
                        raw_body = await response.text()
                        return json.loads(raw_body)

                    else:
                        try:
                            raw_body = await response.text()
                            safe_body_summary = _safe_error_body_summary(raw_body)
                        except Exception:
                            safe_body_summary = "unavailable"
                        error = ClientResponseError(
                            response.request_info,
                            response.history,
                            status=response.status,
                            message=response.reason,
                        )
                        error.safe_response_body = safe_body_summary
                        raise error

        except TimeoutError:
            raise RequestTimeoutError(f"Request timed out after {TIMEOUT}s: {url}")

    async def get(self, url: str, raw_reply: bool = False, raw_contents: bool = False, **kwargs: Any) -> Any:
        headers = self._get_headers()
        return await self.request(
            METH_GET, url, data=None, headers=headers,
            raw_reply=raw_reply, raw_contents=raw_contents, **kwargs,
        )

    def _get_headers(self) -> dict[str, str]:
        data = {
            "Accept": "application/json",
            "Accept-Charset": "utf-8",
            "X-App-Version": self.HDR_XAPP_VERSION,
            "X-App-Name": "myAudi",
            "User-Agent": self.HDR_USER_AGENT,
        }
        if self._token is not None:
            data["Authorization"] = "Bearer " + self._token.get("access_token")
        if self._xclient_id is not None:
            data["X-Client-ID"] = self._xclient_id
        return data
