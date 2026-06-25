"""Tests for CVE fetching and storage."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from app.core.config import CveConfig
from app.cve import CveRecord, CveService, CveStore, NvdCveClient


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def _nvd_payload(cve_id: str = "CVE-2026-1234") -> dict:
    return {
        "vulnerabilities": [
            {
                "cve": {
                    "id": cve_id,
                    "sourceIdentifier": "security@example.test",
                    "published": "2026-06-25T10:00:00.000",
                    "lastModified": "2026-06-25T11:00:00.000",
                    "vulnStatus": "Analyzed",
                    "descriptions": [
                        {"lang": "en", "value": "Example vulnerability."}
                    ],
                    "metrics": {
                        "cvssMetricV31": [
                            {
                                "cvssData": {
                                    "version": "3.1",
                                    "baseScore": 9.8,
                                    "baseSeverity": "CRITICAL",
                                }
                            }
                        ]
                    },
                    "weaknesses": [
                        {"description": [{"lang": "en", "value": "CWE-79"}]}
                    ],
                    "references": {
                        "referenceData": [{"url": "https://example.test/advisory"}]
                    },
                }
            }
        ]
    }


def test_nvd_client_fetches_and_normalizes_latest_cves() -> None:
    seen = {}

    def opener(request, timeout):
        seen["timeout"] = timeout
        seen["query"] = parse_qs(urlparse(request.full_url).query)
        return FakeResponse(_nvd_payload())

    client = NvdCveClient(
        "https://services.nvd.nist.gov/rest/json/cves/2.0",
        timeout_seconds=7,
        opener=opener,
    )

    records = client.fetch_latest(hours=12, limit=25)

    assert seen["timeout"] == 7
    assert seen["query"]["resultsPerPage"] == ["25"]
    assert "pubStartDate" in seen["query"]
    assert records[0].cve_id == "CVE-2026-1234"
    assert records[0].severity == "CRITICAL"
    assert records[0].base_score == 9.8
    assert records[0].weaknesses == ("CWE-79",)
    assert records[0].references == ("https://example.test/advisory",)


def test_cve_store_upserts_by_id_and_sorts_newest_first(tmp_path: Path) -> None:
    store = CveStore(tmp_path / "cves.json", max_items=10)
    older = CveRecord(
        cve_id="CVE-2026-0001",
        published="2026-06-24T10:00:00.000",
        last_modified="2026-06-24T10:00:00.000",
        status="Analyzed",
        source_identifier="source",
        description="older",
    )
    newer = CveRecord(
        cve_id="CVE-2026-0002",
        published="2026-06-25T10:00:00.000",
        last_modified="2026-06-25T10:00:00.000",
        status="Analyzed",
        source_identifier="source",
        description="newer",
    )
    updated = CveRecord(
        cve_id="CVE-2026-0001",
        published="2026-06-26T10:00:00.000",
        last_modified="2026-06-26T10:00:00.000",
        status="Modified",
        source_identifier="source",
        description="updated",
    )

    store.add_many([older, newer])
    records = store.add_many([updated])

    assert [record.cve_id for record in records] == ["CVE-2026-0001", "CVE-2026-0002"]
    assert records[0].description == "updated"
    assert store.get("cve-2026-0001") == records[0]


def test_cve_service_fetches_and_persists_with_config_defaults(tmp_path: Path) -> None:
    class FakeClient:
        def fetch_latest(self, *, hours: int, limit: int):
            assert hours == 24
            assert limit == 50
            return [
                CveRecord(
                    cve_id="CVE-2026-9999",
                    published="2026-06-25T10:00:00.000",
                    last_modified="2026-06-25T10:00:00.000",
                    status="Received",
                    source_identifier="source",
                    description="stored",
                )
            ]

    config = CveConfig(storage_path=str(tmp_path / "cves.json"))
    service = CveService(config, client=FakeClient())

    records = service.fetch_latest()

    assert records[0].cve_id == "CVE-2026-9999"
    assert service.get("CVE-2026-9999") is not None
