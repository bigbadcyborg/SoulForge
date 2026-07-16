"""Microphone capture + transcription for the GUI (Windows).

Toggle model: the first hotkey press starts recording, the second stops and
uploads the clip to the WSL /transcribe endpoint (robust across hotkey backends,
which only reliably report key-press). Recording uses sounddevice at 16 kHz mono
(what Whisper expects); the WAV encoding lives in ``gui.util`` for testability.
Requires PySide6 + sounddevice (Windows GUI venv).
"""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from gui.api_client import ApiClient
from gui.util import frames_to_wav_bytes

SAMPLE_RATE = 16000


class MicRecorder:
    """Records mono float32 frames from the default input device."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self._stream = None
        self._frames: list = []

    @property
    def recording(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        import sounddevice as sd

        self._frames = []

        def callback(indata, frames, time, status):  # noqa: ANN001
            self._frames.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> bytes:
        if self._stream is None:
            return frames_to_wav_bytes([], self.sample_rate)
        self._stream.stop()
        self._stream.close()
        self._stream = None
        return frames_to_wav_bytes(self._frames, self.sample_rate)


class TranscribeWorker(QThread):
    """Uploads a recorded WAV to the transcription endpoint off the UI thread."""

    done = Signal(str)
    error = Signal(str)

    def __init__(self, client: ApiClient, wav_bytes: bytes) -> None:
        super().__init__()
        self._client = client
        self._wav = wav_bytes

    def run(self) -> None:
        try:
            result = self._client.transcribe(self._wav)
            self.done.emit(result.get("text", ""))
        except Exception as error:  # noqa: BLE001
            self.error.emit(f"Transcription failed: {error}")
