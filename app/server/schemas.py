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
    ready: bool  # chat model loaded
    compute_backend: str
    stage: str = "ready"
    vision_loaded: bool = False
    loading: bool = False  # a session load (chat/agents/vision) is still running


class SessionStartRequest(BaseModel):
    chat_model: str | None = None  # None = keep configured default
    load_agents: bool = False
    load_vision: bool = False


class SessionStartResponse(BaseModel):
    started: bool
    message: str = ""


class SnapshotResponse(BaseModel):
    text: str
    success: bool = True


class TranscribeResponse(BaseModel):
    text: str
    success: bool = True
