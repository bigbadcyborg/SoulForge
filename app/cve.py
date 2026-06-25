"""Fetch and store CVEs from the NVD API."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.core.config import CveConfig

UrlOpen = Callable[..., Any]


@dataclass(frozen=True)
class CveRecord:
    cve_id: str
    published: str
    last_modified: str
    status: str
    source_identifier: str
    description: str
    severity: str | None = None
    base_score: float | None = None
    cvss_version: str | None = None
    weaknesses: tuple[str, ...] = ()
    references: tuple[str, ...] = ()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _nvd_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _english_description(cve: dict[str, Any]) -> str:
    for item in cve.get("descriptions", []):
        if item.get("lang") == "en" and item.get("value"):
            return str(item["value"])
    return ""


def _extract_cvss(cve: dict[str, Any]) -> tuple[str | None, float | None, str | None]:
    metrics = cve.get("metrics", {})
    for key in ("cvssMetricV40", "cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        entries = metrics.get(key) or []
        if not entries:
            continue
        metric = entries[0]
        cvss_data = metric.get("cvssData", {})
        severity = metric.get("baseSeverity") or cvss_data.get("baseSeverity")
        score = cvss_data.get("baseScore")
        version = cvss_data.get("version")
        return severity, score, version
    return None, None, None


def _extract_weaknesses(cve: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for weakness in cve.get("weaknesses", []):
        for description in weakness.get("description", []):
            value = description.get("value")
            if value and value not in values:
                values.append(str(value))
    return tuple(values)


def _extract_references(cve: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for reference in cve.get("references", {}).get("referenceData", []):
        url = reference.get("url")
        if url and url not in values:
            values.append(str(url))
    return tuple(values)


def record_from_nvd_vulnerability(vulnerability: dict[str, Any]) -> CveRecord:
    cve = vulnerability.get("cve", {})
    severity, score, version = _extract_cvss(cve)
    return CveRecord(
        cve_id=str(cve.get("id", "")),
        published=str(cve.get("published", "")),
        last_modified=str(cve.get("lastModified", "")),
        status=str(cve.get("vulnStatus", "")),
        source_identifier=str(cve.get("sourceIdentifier", "")),
        description=_english_description(cve),
        severity=severity,
        base_score=score,
        cvss_version=version,
        weaknesses=_extract_weaknesses(cve),
        references=_extract_references(cve),
    )


class NvdCveClient:
    """Small NVD API client with injectable transport for tests."""

    def __init__(
        self,
        source_url: str,
        timeout_seconds: int = 20,
        opener: UrlOpen = urlopen,
    ) -> None:
        self.source_url = source_url
        self.timeout_seconds = timeout_seconds
        self.opener = opener

    def fetch_latest(self, *, hours: int, limit: int) -> list[CveRecord]:
        end = _utc_now()
        start = end - timedelta(hours=max(hours, 1))
        params = {
            "pubStartDate": _nvd_timestamp(start),
            "pubEndDate": _nvd_timestamp(end),
            "resultsPerPage": max(1, min(limit, 2000)),
        }
        request = Request(
            f"{self.source_url}?{urlencode(params)}",
            headers={
                "Accept": "application/json",
                "User-Agent": "SoulForge-CVE-fetcher/1.0",
            },
        )
        with self.opener(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))

        records = [
            record_from_nvd_vulnerability(item)
            for item in payload.get("vulnerabilities", [])
        ]
        return [record for record in records if record.cve_id]


class CveStore:
    """JSON-backed CVE list with ID-based upserts."""

    def __init__(self, path: Path, max_items: int = 500) -> None:
        self.path = path
        self.max_items = max_items

    def list(self) -> list[CveRecord]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return [
            CveRecord(
                cve_id=item["cve_id"],
                published=item.get("published", ""),
                last_modified=item.get("last_modified", ""),
                status=item.get("status", ""),
                source_identifier=item.get("source_identifier", ""),
                description=item.get("description", ""),
                severity=item.get("severity"),
                base_score=item.get("base_score"),
                cvss_version=item.get("cvss_version"),
                weaknesses=tuple(item.get("weaknesses", [])),
                references=tuple(item.get("references", [])),
            )
            for item in payload.get("cves", [])
        ]

    def save(self, records: list[CveRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        records = sorted(records, key=lambda item: item.published, reverse=True)
        records = records[: self.max_items]
        payload = {
            "updated_at": _utc_now().isoformat(),
            "cves": [asdict(record) for record in records],
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def add_many(self, records: list[CveRecord]) -> list[CveRecord]:
        by_id = {record.cve_id: record for record in self.list()}
        for record in records:
            by_id[record.cve_id] = record
        merged = list(by_id.values())
        self.save(merged)
        return self.list()

    def get(self, cve_id: str) -> CveRecord | None:
        normalized = cve_id.upper()
        for record in self.list():
            if record.cve_id.upper() == normalized:
                return record
        return None


class CveService:
    def __init__(self, config: CveConfig, client: NvdCveClient | None = None) -> None:
        self.config = config
        self.client = client or NvdCveClient(
            config.source_url,
            timeout_seconds=config.request_timeout_seconds,
        )
        self.store = CveStore(config.storage_file, max_items=config.max_items)

    def list(self, *, limit: int | None = None) -> list[CveRecord]:
        records = self.store.list()
        if limit is None:
            return records
        return records[: max(1, limit)]

    def get(self, cve_id: str) -> CveRecord | None:
        return self.store.get(cve_id)

    def fetch_latest(
        self,
        *,
        hours: int | None = None,
        limit: int | None = None,
    ) -> list[CveRecord]:
        fetch_hours = hours or self.config.default_hours
        fetch_limit = limit or self.config.default_limit
        fetched = self.client.fetch_latest(hours=fetch_hours, limit=fetch_limit)
        return self.store.add_many(fetched)
