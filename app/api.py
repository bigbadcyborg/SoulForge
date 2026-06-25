"""HTTP API for SoulForge CVE fetching."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

from app.core.config import load_config
from app.cve import CveRecord, CveService


def _serialize(record: CveRecord) -> dict:
    payload = asdict(record)
    payload["weaknesses"] = list(record.weaknesses)
    payload["references"] = list(record.references)
    return payload


def create_app(config_path: str | Path | None = None) -> FastAPI:
    config = load_config(config_path)
    service = CveService(config.cve)
    app = FastAPI(
        title="SoulForge CVE API",
        description="Fetch latest CVEs from NVD and expose a local CVE list.",
        version="1.0.0",
    )

    @app.get("/")
    def root() -> dict:
        return {
            "service": "SoulForge CVE API",
            "docs": "/docs",
            "endpoints": ["/health", "/cves", "/cves/fetch", "/cves/{cve_id}"],
        }

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "stored_cves": len(service.list()),
            "source_url": config.cve.source_url,
        }

    @app.get("/cves")
    def list_cves(
        limit: int = Query(50, ge=1, le=500),
        severity: str | None = Query(None),
    ) -> dict:
        records = service.list(limit=limit)
        if severity:
            expected = severity.upper()
            records = [
                record
                for record in records
                if (record.severity or "").upper() == expected
            ]
        return {"count": len(records), "cves": [_serialize(record) for record in records]}

    @app.get("/cves/{cve_id}")
    def get_cve(cve_id: str) -> dict:
        record = service.get(cve_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"CVE not found: {cve_id}")
        return _serialize(record)

    @app.post("/cves/fetch")
    def fetch_cves(
        hours: int | None = Query(None, ge=1, le=120),
        limit: int | None = Query(None, ge=1, le=2000),
    ) -> dict:
        records = service.fetch_latest(hours=hours, limit=limit)
        return {
            "count": len(records),
            "cves": [_serialize(record) for record in records],
        }

    return app


app = create_app()
