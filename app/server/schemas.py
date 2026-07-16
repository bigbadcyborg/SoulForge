"""Pydantic request/response models for the SoulForge API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class CommandRequest(BaseModel):
    name: str
    args: str = ""


class CommandResponse(BaseModel):
    kind: str
    text: str = ""
    data: dict[str, Any] = {}
    success: bool = True


class PingResponse(BaseModel):
    status: str
    model: str
    ready: bool
    compute_backend: str


class SnapshotResponse(BaseModel):
    text: str
    success: bool = True


class TranscribeResponse(BaseModel):
    text: str
    success: bool = True
