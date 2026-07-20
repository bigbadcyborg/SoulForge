"""FastAPI application wrapping ChatController for the desktop GUI.

Blocking controller calls run in a worker thread so the event loop stays
responsive (the server analog of the TUI's ``@work`` threads). Streaming chat
is delivered over a WebSocket: a worker thread drives the synchronous
``add_user_turn -> stream_reply -> finalize_assistant_reply -> complete_turn``
sequence and pushes JSON frames back onto the event loop.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from starlette.concurrency import run_in_threadpool

from app.core.chat_controller import ChatController
from app.core.command_router import CommandRouter
from app.server.schemas import (
    CommandRequest,
    CommandResponse,
    PingResponse,
    SessionStartRequest,
    SessionStartResponse,
    SnapshotResponse,
    TranscribeResponse,
)

_SENTINEL = object()


def _require_token(expected: str):
    """Build a dependency enforcing the shared secret when one is configured."""

    async def _dep(x_soulforge_token: str | None = Header(default=None)) -> None:
        if expected and x_soulforge_token != expected:
            raise HTTPException(status_code=401, detail="Invalid or missing token.")

    return _dep


def _chat_frames(controller: ChatController, text: str, loop, queue: asyncio.Queue) -> None:
    """Run one generation turn on a worker thread, pushing frames to the queue."""

    def emit(frame: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, frame)

    try:
        chunks = controller.add_user_turn(text)
        if controller.features.is_enabled("show_sources") and chunks:
            emit(
                {
                    "type": "sources",
                    "sources": [
                        {"source": c.source, "chunk_index": c.chunk_index}
                        for c in chunks
                    ],
                }
            )
        parts: list[str] = []
        if controller.features.is_enabled("streaming"):
            for token in controller.stream_reply():
                parts.append(token)
                emit({"type": "token", "token": token})
            raw_reply = getattr(controller, "_pending_raw_reply", "") or "".join(parts)
        else:
            raw_reply = controller.full_reply()
            emit({"type": "token", "token": raw_reply})

        tool_turn = controller.finalize_assistant_reply(raw_reply)
        emit({"type": "final", "text": tool_turn.display_text})
        if tool_turn.has_pending:
            emit(
                {
                    "type": "tool",
                    "pending": [
                        controller.format_tool_call_preview(p)
                        for p in tool_turn.pending
                    ],
                }
            )

        review = controller.complete_turn()
        if review.message:
            emit({"type": "review", "text": review.message})
    except Exception as error:  # noqa: BLE001 - surface to the client
        emit({"type": "error", "text": f"Generation error: {error}"})
    finally:
        loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)


def run_session_load(controller: ChatController, request, state: dict) -> None:
    """Load the chosen models for a session, updating ``state`` as it goes.

    Runs on a worker thread. ``request`` has chat_model / load_agents /
    load_vision. Best-effort: a vision/agent failure is recorded but does not
    stop the chat model from becoming ready.
    """
    # Surface the exact model being loaded (e.g. "Loading agent model 'creator'")
    # to ping via the shared state, so the GUI shows real per-model progress.
    controller.runtime.set_load_listener(lambda msg: state.__setitem__("stage", msg))
    try:
        state["stage"] = "Loading chat model..."
        chat_model = getattr(request, "chat_model", None)
        current = controller.model_name
        if chat_model and chat_model not in ("", current):
            controller.switch_chat_model(chat_model)
        elif not controller.loaded:
            controller.load()  # skip if the chat model is already loaded

        # RAG and episodic memory embed every prompt, so preload the embedding
        # model too — otherwise the first prompt stalls loading it lazily.
        if controller.features.is_enabled("memory") or controller.features.is_enabled("rag"):
            try:
                controller.runtime.load_embedding_model()
            except Exception as error:  # noqa: BLE001
                state["error"] = f"embedding preload failed: {error}"

        if getattr(request, "load_agents", False):
            try:
                controller.runtime.warm_resident_profiles()
            except Exception as error:  # noqa: BLE001
                state["error"] = f"agent preload failed: {error}"

        if getattr(request, "load_vision", False) and controller.config.vision.enabled:
            try:
                if not controller.runtime.vision_loaded:
                    controller.runtime.preload_vision_model()
                state["vision_loaded"] = True
            except Exception as error:  # noqa: BLE001
                state["error"] = f"vision preload failed: {error}"
    except Exception as error:  # noqa: BLE001
        state["error"] = f"model load failed: {error}"
    finally:
        controller.runtime.set_load_listener(None)
        state["stage"] = "ready"
        state["loading"] = False


def create_app(controller: ChatController, transcriber=None) -> FastAPI:
    """Build the FastAPI app bound to a ready ChatController.

    ``transcriber`` is injectable (tests pass a stub); when omitted a lazy
    faster-whisper ``Transcriber`` is created — it loads no model until the
    first /transcribe call.
    """
    app = FastAPI(title="SoulForge API", version="1.0")
    router = CommandRouter(controller)
    auth = _require_token(controller.config.server.auth_token)

    if transcriber is None:
        from app.server.transcribe import Transcriber

        transcriber = Transcriber(controller.config.transcription)

    # Shared session-load state (updated by the loader thread, read by ping).
    session_state: dict = {"stage": "idle", "vision_loaded": False, "loading": False}

    @app.get("/api/ping", response_model=PingResponse)
    async def ping() -> PingResponse:
        stage = session_state["stage"]
        if stage in ("idle", "ready"):
            stage = "ready" if controller.loaded else "idle"
        return PingResponse(
            status="ok",
            model=controller.model_name,
            ready=controller.loaded,
            compute_backend=str(controller.compute_backend),
            stage=stage,
            vision_loaded=session_state["vision_loaded"] or controller.runtime.vision_loaded,
            loading=session_state["loading"],
        )

    @app.post(
        "/api/session/start",
        response_model=SessionStartResponse,
        dependencies=[Depends(auth)],
    )
    async def session_start(request: SessionStartRequest) -> SessionStartResponse:
        if session_state["loading"]:
            return SessionStartResponse(started=False, message="Already loading.")
        session_state.update({"loading": True, "stage": "starting", "error": None})
        threading.Thread(
            target=run_session_load,
            args=(controller, request, session_state),
            daemon=True,
        ).start()
        return SessionStartResponse(started=True, message="Loading started.")

    @app.get("/api/commands", dependencies=[Depends(auth)])
    async def commands() -> dict[str, list[str]]:
        return {"commands": router.command_names()}

    @app.post("/api/command", response_model=CommandResponse, dependencies=[Depends(auth)])
    async def command(request: CommandRequest) -> CommandResponse:
        # Router calls into blocking ChatController methods; keep them off-loop.
        result = await run_in_threadpool(router.dispatch, request.name, request.args)
        return CommandResponse(**result.to_dict())

    @app.post("/api/snapshot", response_model=SnapshotResponse, dependencies=[Depends(auth)])
    async def snapshot(
        image: UploadFile = File(...),
        prompt: str = Form(default=""),
        inject: bool = Form(default=True),
    ) -> SnapshotResponse:
        if not controller.config.vision.enabled:
            raise HTTPException(
                status_code=400,
                detail="Vision model not configured (set vision.modelPath).",
            )
        data = await image.read()

        def run() -> str:
            answer = controller.runtime.create_vision_completion(data, prompt)
            if inject:
                # Keep the model conversation coherent for follow-up questions.
                label = prompt or "Describe this image."
                controller.messages.append(
                    {"role": "user", "content": f"[Screen snapshot] {label}"}
                )
                controller.messages.append({"role": "assistant", "content": answer})
            return answer

        answer = await run_in_threadpool(run)
        return SnapshotResponse(text=answer)

    @app.post(
        "/api/transcribe",
        response_model=TranscribeResponse,
        dependencies=[Depends(auth)],
    )
    async def transcribe(
        audio: UploadFile = File(...),
        language: str = Form(default=""),
    ) -> TranscribeResponse:
        data = await audio.read()
        text = await run_in_threadpool(transcriber.transcribe_wav, data, language)
        return TranscribeResponse(text=text)

    @app.post("/api/rag/upload", response_model=CommandResponse, dependencies=[Depends(auth)])
    async def rag_upload(document: UploadFile = File(...)) -> CommandResponse:
        data = await document.read()
        text = await run_in_threadpool(
            controller.save_uploaded_doc, document.filename or "document", data
        )
        return CommandResponse(kind="message", text=text, success=True)

    @app.websocket("/ws/chat")
    async def chat(websocket: WebSocket) -> None:
        token = controller.config.server.auth_token
        if token and websocket.query_params.get("token") != token:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        loop = asyncio.get_running_loop()
        try:
            while True:
                payload = await websocket.receive_json()
                text = str(payload.get("message", "")).strip()
                if not text:
                    await websocket.send_json({"type": "error", "text": "Empty message."})
                    continue
                if not controller.loaded or session_state["loading"]:
                    # A background load holds the model lock; refuse rather than
                    # hang the prompt until it finishes.
                    await websocket.send_json(
                        {"type": "error", "text": "Models still loading — please wait."}
                    )
                    continue
                queue: asyncio.Queue = asyncio.Queue()
                threading.Thread(
                    target=_chat_frames,
                    args=(controller, text, loop, queue),
                    daemon=True,
                ).start()
                while True:
                    frame = await queue.get()
                    if frame is _SENTINEL:
                        break
                    await websocket.send_json(frame)
                await websocket.send_json({"type": "done"})
        except WebSocketDisconnect:
            return

    return app
