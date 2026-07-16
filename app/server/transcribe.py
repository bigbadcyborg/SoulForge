"""Local speech-to-text with faster-whisper (runs on the WSL GPU).

The model is loaded lazily and cached on first use so importing this module (and
the server) never pulls the heavy dependency until a transcription is requested.
"""

from __future__ import annotations

import io
import threading

from app.core.config import TranscriptionConfig


class Transcriber:
    """Lazy, thread-safe wrapper around a faster-whisper model."""

    def __init__(self, config: TranscriptionConfig) -> None:
        self.config = config
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(
                    self.config.model_size,
                    device=self.config.device,
                    compute_type=self.config.compute_type,
                )
        return self._model

    def transcribe_wav(self, wav_bytes: bytes, language: str = "") -> str:
        """Transcribe WAV/audio bytes to text; empty language autodetects."""
        model = self._get_model()
        lang = language or self.config.language or None
        segments, _info = model.transcribe(io.BytesIO(wav_bytes), language=lang)
        return " ".join(segment.text.strip() for segment in segments).strip()
