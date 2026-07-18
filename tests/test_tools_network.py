"""Tests for the sandboxed fetch_url network tool."""

from __future__ import annotations

import pytest

from app.core.config import ToolsConfig
from app.tools.handlers import net


class _FakeConfig:
    def __init__(self, tools: ToolsConfig) -> None:
        self.tools = tools


def _config(*, allow: bool = True, allowlist=None) -> _FakeConfig:
    return _FakeConfig(
        ToolsConfig(
            allow_network=allow,
            network_allowlist=allowlist if allowlist is not None else ["example.com"],
        )
    )


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200, headers=None) -> None:
        self._body = body
        self.status = status
        self.headers = _FakeHeaders(headers or {})

    def read(self, amt=None):
        if amt is None:
            return self._body
        return self._body[:amt]

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeHeaders:
    def __init__(self, mapping):
        self._mapping = mapping

    def get(self, key, default=""):
        return self._mapping.get(key, default)

    def get_content_charset(self):
        return self._mapping.get("charset", "utf-8")


def test_fetch_url_requires_allownetwork() -> None:
    with pytest.raises(PermissionError, match="allowNetwork"):
        net.fetch_url(_config(allow=False), {"url": "https://example.com"})


def test_fetch_url_rejects_non_allowlisted_domain() -> None:
    with pytest.raises(PermissionError, match="networkAllowlist"):
        net.fetch_url(_config(allowlist=["example.com"]), {"url": "https://evil.com"})


def test_fetch_url_rejects_non_http_scheme() -> None:
    with pytest.raises(PermissionError, match="http/https"):
        net.fetch_url(_config(), {"url": "file:///etc/passwd"})


def test_fetch_url_rejects_private_ip(monkeypatch) -> None:
    # Host is allowlisted but resolves to a loopback address, which is refused.
    monkeypatch.setattr(
        net.socket,
        "getaddrinfo",
        lambda host, port: [(None, None, None, "", ("127.0.0.1", 0))],
    )
    with pytest.raises(PermissionError, match="non-public"):
        net.fetch_url(
            _config(allowlist=["internal.example.com"]),
            {"url": "https://internal.example.com"},
        )


def test_fetch_url_returns_body(monkeypatch) -> None:
    monkeypatch.setattr(
        net.socket,
        "getaddrinfo",
        lambda host, port: [(None, None, None, "", ("93.184.216.34", 0))],
    )
    monkeypatch.setattr(
        net._opener,
        "open",
        lambda request, timeout=None: _FakeResponse(b"hello world"),
    )
    out = net.fetch_url(_config(), {"url": "https://example.com"})
    assert out == "hello world"


def test_fetch_url_enforces_size_cap(monkeypatch) -> None:
    monkeypatch.setattr(
        net.socket,
        "getaddrinfo",
        lambda host, port: [(None, None, None, "", ("93.184.216.34", 0))],
    )
    monkeypatch.setattr(
        net._opener,
        "open",
        lambda request, timeout=None: _FakeResponse(b"abcdefghij"),
    )
    out = net.fetch_url(_config(), {"url": "https://example.com", "max_bytes": 4})
    assert out.startswith("abcd")
    assert "truncated" in out


def test_fetch_url_revalidates_redirect_target(monkeypatch) -> None:
    monkeypatch.setattr(
        net.socket,
        "getaddrinfo",
        lambda host, port: [(None, None, None, "", ("93.184.216.34", 0))],
    )
    # A redirect toward a non-allowlisted host must be refused, not followed.
    redirect = _FakeResponse(
        b"", status=302, headers={"Location": "https://evil.com/x"}
    )
    monkeypatch.setattr(net._opener, "open", lambda request, timeout=None: redirect)
    with pytest.raises(PermissionError, match="networkAllowlist"):
        net.fetch_url(_config(allowlist=["example.com"]), {"url": "https://example.com"})
