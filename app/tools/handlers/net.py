"""Network tool handlers.

A single read-only HTTP GET tool with a deny-by-default sandbox:

* only ``http``/``https`` URLs,
* host must be on ``tools.networkAllowlist`` (checked in permissions),
* the resolved IP must be public (no loopback/private/link-local addresses),
* hard timeout and response-size cap,
* redirects are re-validated against the same rules.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.error import URLError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from app.core.config import AppConfig
from app.tools.permissions import check_network_host

_DEFAULT_MAX_BYTES = 100_000
_TIMEOUT_SECONDS = 10
_MAX_REDIRECTS = 3


class _NoAutoRedirect(HTTPRedirectHandler):
    """Return 3xx responses instead of silently following them.

    Automatic redirects would bypass the per-hop allowlist and public-IP checks,
    so we handle each redirect manually.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None


_opener = build_opener(_NoAutoRedirect)


def _assert_public_host(host: str) -> None:
    """Reject hosts that resolve to loopback/private/link-local/reserved IPs."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as error:
        raise PermissionError(f"Could not resolve host: {host} ({error})") from error
    for info in infos:
        address = info[4][0]
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise PermissionError(
                f"Host {host} resolves to a non-public address ({address})"
            )


def _validate_url(config: AppConfig, url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise PermissionError("fetch_url only supports http/https URLs")
    host = parsed.hostname or ""
    check_network_host(config, host)  # allowlist + allowNetwork gate
    _assert_public_host(host)  # only allow public addresses
    return url


def fetch_url(config: AppConfig, args: dict) -> str:
    url = str(args.get("url", "")).strip()
    if not url:
        raise ValueError("fetch_url requires url")

    try:
        max_bytes = int(args.get("max_bytes", _DEFAULT_MAX_BYTES))
    except (TypeError, ValueError):
        max_bytes = _DEFAULT_MAX_BYTES
    max_bytes = max(1, min(max_bytes, _DEFAULT_MAX_BYTES))

    current = _validate_url(config, url)
    for _ in range(_MAX_REDIRECTS + 1):
        request = Request(current, headers={"User-Agent": "SoulForge/1.0"}, method="GET")
        try:
            with _opener.open(request, timeout=_TIMEOUT_SECONDS) as response:  # noqa: S310
                status = getattr(response, "status", None) or response.getcode()
                if status in (301, 302, 303, 307, 308):
                    location = response.headers.get("Location", "")
                    if not location:
                        raise PermissionError("Redirect without Location header")
                    current = _validate_url(config, urljoin(current, location))
                    continue
                raw = response.read(max_bytes + 1)
                charset = response.headers.get_content_charset() or "utf-8"
        except URLError as error:
            raise RuntimeError(f"fetch_url failed: {error.reason}") from error

        text = raw.decode(charset, errors="replace")
        if len(raw) > max_bytes:
            text = text[:max_bytes] + f"\n\n[truncated at {max_bytes} bytes]"
        return text

    raise PermissionError("Too many redirects")
